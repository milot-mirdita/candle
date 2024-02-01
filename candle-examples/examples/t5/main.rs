#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::path::PathBuf;

use candle_transformers::models::t5::{self, T5EncoderModel};

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor, D, IndexOp};
use candle_nn::{Module, VarBuilder, Conv2d, Conv2dConfig,  Activation, Dropout};
use candle_transformers::generation::LogitsProcessor;
//use crate::models::model_utils::{Dropout, HiddenAct, Linear, HiddenActLayer, LayerNorm, PositionEmbeddingType};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
//use tokenizers::Tokenizer;

use serde_json; 
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Write, Read};
use std::path::Path;

// use rust_tokenizers::tokenizer::{
//     T5Tokenizer, Tokenizer, TruncationStrategy
// };

const DTYPE: DType = DType::F16;

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model repository to use on the HuggingFace hub.
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// Enable decoding.
    #[arg(long)]
    decode: bool,

    // Enable/disable decoding.
    #[arg(long, default_value = "false")]
    disable_cache: bool,

    /// Use this prompt, otherwise compute sentence similarities.
    #[arg(long)]
    prompt: Option<String>,

    /// If set along with --decode, will use this prompt to initialize the decoder.
    #[arg(long)]
    decoder_prompt: Option<String>,

    /// L2 normalization for embeddings.
    #[arg(long, default_value = "true")]
    normalize_embeddings: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 64)]
    repeat_last_n: usize,
}


fn process_fasta_batch(input_path: &Path, output_path: &str, model: &mut T5EncoderModel, cnn: &CNN,  hashmap:&HashMap<String,usize>, device: &Device, max_residues: i32, max_batch: i32) -> io::Result<()> {
    let file = File::open(input_path)?;
    let reader = io::BufReader::new(file);
    let mut output_file = OpenOptions::new().create(true).write(true).truncate(true).open(output_path)?;

    let mut s = String::new();
    let mut current_residues = 0;
    let mut current_batch = 0;
    let mut batch: Vec<String> = Vec::new();
    let mut batch_ids: Vec<String> = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            batch_ids.push(line.clone());
        } 
        else if (current_residues + line.len() as i32 >= max_residues || current_batch + 1 >= max_batch) {
                let strings = batch.clone();
                let res = predict_batch(strings, model, cnn,  hashmap, device);
                let seqs = res.unwrap();
                for i in 1..seqs.len(){
                    writeln!(output_file, "{}", batch_ids[i])?;
                    writeln!(output_file, "{}", seqs[i])?;
                }
                // for seq in seqs{
                //     println!("{seq}");
                // }
                batch.clear();
                let n = batch_ids.len();
                let batch_ids = vec![batch_ids[n-1].clone()];
                current_residues = 0;
                current_batch = 0;
        } else {
            batch.push(line.clone());
            current_residues += line.len() as i32;
            current_batch += 1;
        }
    }
    let strings = batch.clone();
    let res = predict_batch(strings, model, cnn,  hashmap, device);
    let seqs = res.unwrap();
    for i in 1..seqs.len(){
        writeln!(output_file, "{}", batch_ids[i])?;
        writeln!(output_file, "{}", seqs[i])?;
    }
    Ok(())
}

fn predict_batch(strings: Vec<String>, model: &mut T5EncoderModel, cnn: &CNN, hashmap:&HashMap<String,usize>, device: &Device) -> Result<Vec<String>>{

    let mut structures: Vec<String> = Vec::new();
    let mut all_tokens: Vec<Vec<i64>> = Vec::new();
    for prompt in strings.iter(){
        let copy = &prompt;

        // Replace each character in the string
        let replaced_values: Vec<Option<&usize>> = copy.chars()
            .map(|c| hashmap.get(&c.to_string()))
            .collect();
        
        let unknown_value: usize = 2; // Default value for None

        let ts: Vec<&usize> = replaced_values
            .iter()
            .map(|option| option.unwrap_or(&unknown_value))
            .collect();
        let mut tokens: Vec<&usize> = vec![hashmap.get("<AA2fold>").unwrap()];
        tokens.extend(ts.iter().clone());
        tokens.push(hashmap.get("</s>").unwrap());
        let tokens: Vec<i64> = tokens
        .iter()
        .map(|&num| *num as i64)
        .collect();
        all_tokens.push(tokens);
    }
    let max_length = all_tokens.iter().map(|x| x.len()).max().unwrap();
    for t in &mut all_tokens{
        t.resize(max_length, 0);
        //t.push(1);
    }
    let input_token_ids = Tensor::new(all_tokens, device)?;
    //println!("{input_token_ids}");
    let ys = model.forward(&input_token_ids)?;
    // FIXME
    let ys = ys.i((.., 1..ys.dims3()?.1-1))?;
    let ys = ys.pad_with_zeros(1, 0, 1)?;

    let output = &cnn.forward(&ys)?;
    println!("{:?}", output.shape());
    
    let vals = output.argmax_keepdim(1)?;
    let vals = vals.flatten(0, 2)?;
    let vals = vals.to_vec1::<u32>()?;
    let structure: Vec<char> = vals.iter().map(|&n| number_to_char(n)).collect();
    let structure: String = structure.into_iter().collect();
    structures.push(structure.clone());

    Ok(structures)
            
}

fn predict(prompt: String, model: &mut T5EncoderModel, cnn: &CNN, hashmap:&HashMap<String,usize>, device: &Device) -> Result<Vec<String>>{

    let copy = &prompt;
    // Replace each character in the string
    let replaced_values: Vec<Option<&usize>> = copy.chars()
        .map(|c| hashmap.get(&c.to_string()))
        .collect();
    
    let unknown_value: usize = 2; // Default value for None

    let ts: Vec<&usize> = replaced_values
        .iter()
        .map(|option| option.unwrap_or(&unknown_value))
        .collect();
    let mut tokens: Vec<&usize> = vec![hashmap.get("<AA2fold>").unwrap()];
    tokens.extend(ts.iter().clone());
    tokens.push(hashmap.get("</s>").unwrap());
    let tokens: Vec<i64> = tokens
    .iter()
    .map(|&num| *num as i64)
    .collect();
    // for value in tokens.iter(){
    //     print!("{:?} ", value);
    // }
    //println!("\n");
    //ts.insert(0, hashmap.get("<AA2FOLD>").unwrap());
    // let tokens=tokenizer.encode(&prompt, None, 1024, &TruncationStrategy::LongestFirst, 0).token_ids;
    // for value in &tokens {
    //     print!("{:?} ", value);
    // }
    // // println!("5");
    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?.to_dtype(DType::U8)?;
    //let input_token_ids=input_token_ids.repeat(2)?;
    //println!("{input_token_ids}");
    // println!("6");
    
    //println!("Not decoding, just generating embeddings");
    // println!("7");
    
    let ys = model.forward(&input_token_ids)?;
    // FIXME
    let ys = ys.i((.., 1..ys.dims3()?.1-1))?;
    let ys = ys.pad_with_zeros(1, 0, 1)?;
    //println!("{ys}");
    // println!("8");
    //println!("{:?}", ys.dims3()?.1);
    //println!("{:?}", ys.shape());
    // let ys_s = ys.unsqueeze(0)?;
    
    // println!("9");
    let output = &cnn.forward(&ys)?;
    println!("{:?}", output.shape());
    // println!("10");
    //println!("{output}");
    let mut structures: Vec<String> = Vec::new();
    let vals = output.argmax_keepdim(1)?;
    let vals = vals.flatten(0, 2)?;
    let vals = vals.to_vec1::<u32>()?;
    let structure: Vec<char> = vals.iter().map(|&n| number_to_char(n)).collect();
    let structure: String = structure.into_iter().collect();
    structures.push(structure.clone());
    //     let structure: String = structure.into_iter().collect();
    //     structures.push(structure.clone());
    // for i in 0..output.dims3()?.0{
    //     let output_vec=output.to_vec3::<f32>()?;
    //     let output_matrix = &output_vec[i];
    //     let output_matrix = transpose(output_matrix);
    //     let mut max_indices = Vec::new();

    //     for row in output_matrix {
    //         let max_index = row.iter().enumerate()
    //             .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
    //             .map(|(idx, _)| idx)
    //             .unwrap_or(0); // Default to 0 if row is empty

    //         max_indices.push(max_index);
    //     }
    //     // Print the vector of max indices
    //     //println!("{:?}", max_indices);
    //     //println!("{:?}", max_indices.len());
    //     let structure: Vec<char> = max_indices.iter().map(|&n| number_to_char(n)).collect();
    //     let structure: String = structure.into_iter().collect();
    //     structures.push(structure.clone());
    //     println!("{:?}", structure);
    // }
    //println!("{:?}", ys.shape());
    Ok(structures)
            
}

fn process_fasta(input_path: &Path, output_path: &str, model: &mut T5EncoderModel, cnn: &CNN,  hashmap:&HashMap<String,usize>, device: &Device) -> io::Result<()> {
    let file = File::open(input_path)?;
    let reader = io::BufReader::new(file);
    let mut output_file = OpenOptions::new().create(true).write(true).truncate(true).open(output_path)?;

    let mut s = String::new();

    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if !s.is_empty() {
                let prompt = s.clone();
                let res = predict(prompt, model, cnn,  hashmap, device);
                for value in &res.unwrap(){

                    writeln!(output_file, "{}", value);
                }
                s.clear();
            }
            writeln!(output_file, "{}", line)?;
        } else {
            s.push_str(&line);
        }
    }

    // Write the last sequence if not empty
    if !s.is_empty() {
        writeln!(output_file, "{}", s)?;
    }

    Ok(())
}

fn transpose(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    if matrix.is_empty() || matrix[0].is_empty() {
        return Vec::new();
    }

    let nrows = matrix.len();
    let ncols = matrix[0].len();

    let mut transposed = vec![vec![0.0; nrows]; ncols];

    for i in 0..nrows {
        for j in 0..ncols {
            transposed[j][i] = matrix[i][j];
        }
    }

    transposed
}

fn number_to_char(n: u32) -> char {
    match n {
        0 => 'A',
        1 => 'C',
        2 => 'D',
        3 => 'E',
        4 => 'F',
        5 => 'G',
        6 => 'H',
        7 => 'I',
        8 => 'K',
        9 => 'L',
        10 => 'M',
        11 => 'N',
        12 => 'P',
        13 => 'Q',
        14 => 'R',
        15 => 'S',
        16 => 'T',
        17 => 'V',
        18 => 'W',
        19 => 'Y',
        _ => 'X', // Default case for numbers not in the list
    }
}

pub fn conv2d_non_square(
    in_channels: usize,
    out_channels: usize,
    kernel_size1: usize,
    kernel_size2: usize,
    cfg: Conv2dConfig,
    vb: crate::VarBuilder,
) -> Result<Conv2d> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints(
        (
            out_channels,
            in_channels / cfg.groups,
            kernel_size1,
            kernel_size2,
        ),
        "weight",
        init_ws,
    )?;
    let bound = 1. / (in_channels as f64).sqrt();
    let init_bs = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_channels, "bias", init_bs)?;
    Ok(Conv2d::new(ws, Some(bs), cfg))
}

pub struct CNN{
    conv1: Conv2d,
    act: Activation,
    dropout: Dropout,
    conv2: Conv2d
}

impl CNN{
    pub fn load(vb: VarBuilder, config: Conv2dConfig) -> Result<Self>{
        let conv1 = conv2d_non_square(1024, 32, 7, 1, config, vb.pp("classifier.0"))?;
        let act = Activation::Relu;
        let dropout = Dropout::new(0.0);
        let conv2 = conv2d_non_square(32, 20, 7, 1, config, vb.pp("classifier.3"))?;
        Ok(Self { conv1, act, dropout, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        //println!("input shape: {:?}", xs.shape());
        let xs: Tensor = xs.permute((0, 2, 1))?.unsqueeze(D::Minus1)?;
        //println!("input after permutation: ");
        //println!("{xs}");
        //println!("{:?}", xs.shape());
        // println!("{:?}", xs.shape());
        let xs: Tensor = xs.pad_with_zeros(2, 3, 3)?;
        // println!("{:?}", xs.shape());
        let xs: Tensor = self.conv1.forward(&xs)?;
        // println!("{:?}", xs.shape());
        // println!("{xs}");
        //println!("{:?}", xs.shape());
        let xs: Tensor = xs.relu()?;
        let xs: Tensor = xs.clone();
        let xs: Tensor = xs.pad_with_zeros(2, 3, 3)?;
        let xs = self.conv2.forward(&xs)?.squeeze(D::Minus1)?;
        Ok(xs)
    }
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
    cnn_filename: Vec<PathBuf>
}

impl T5ModelBuilder {
    pub fn load(args: &Args) -> Result<Self> {
        let device = candle_examples::device(args.cpu)?;
        let default_model = "t5-small".to_string();
        let default_revision = "refs/pr/15".to_string();
        let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
            (Some(model_id), Some(revision)) => (model_id, revision),
            (Some(model_id), None) => (model_id, "main".to_string()),
            (None, Some(revision)) => (default_model, revision),
            (None, None) => (default_model, default_revision),
        };

        let repo = Repo::with_revision(model_id.clone(), RepoType::Model, revision);
        let api = Api::new()?;
        let api = api.repo(repo);
        let config_filename = api.get("config.json")?;
        println!("{:?}", config_filename);
        //let tokenizer_filename = api.get("tokenizer.json")?;
        let weights_filename = if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2"
        {
            candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
        } else {
            vec![api.get("model.safetensors")?]
        };
        let mut path = PathBuf::from("C:/Users/SteineggerLab/.cache/huggingface/hub/models--t5-small/snapshots/df1b051c49625cf57a3d0d8d3863ed4d13564fe4/cnn.safetensors");
        //let cnn_filename = path;
        let cnn_filename = vec![path];
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;

        /*let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        Ok((
            Self {
                device,
                config,
                weights_filename,
            },
            tokenizer,
        ))*/
        // let vocab_path: PathBuf  = PathBuf::from("C:/Users/SteineggerLab/Desktop/Victor/ProstT5/scripts/model/models--Rostlab--ProstT5_fp16/snapshots/07a6547d51de603f1be84fd9f2db4680ee535a86/spiece.model");
        // let mut t5_tokenizer = T5Tokenizer::from_file(vocab_path, false).unwrap();
        // t5_tokenizer.add_tokens(&[
        //     "a",
        //     "l",
        //     "g",
        //     "v",
        //     "s",
        //     "r",
        //     "e",
        //     "d",
        //     "t",
        //     "i",
        //     "p",
        //     "k",
        //     "f",
        //     "q",
        //     "n",
        //     "y",
        //     "m",
        //     "h",
        //     "w",
        //     "c",
        //     "<FOLD2AA>",
        //     "<AA2FOLD>",
        // ]);
        Ok(
            Self {
                device,
                config,
                weights_filename,
                cnn_filename
            }
        )
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        //println!("b1");
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        //println!("b2");
        vb.all_paths();
        //println!("b3");
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }
    pub fn build_CNN(&self) -> Result<CNN> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.cnn_filename, DTYPE, &self.device)?
        };
        //println!("varbuilder initialized!");
        let config = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        Ok(CNN::load(vb,config)?)
    }
    pub fn build_conditional_generation(&self) -> Result<t5::T5ForConditionalGeneration> {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5ForConditionalGeneration::load(vb, &self.config)?)
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    //println!("1");
    let builder = T5ModelBuilder::load(&args)?;
    //println!("2");
    let device = &builder.device;
    let path = Path::new("C:/Users/SteineggerLab/Desktop/Victor/candle/candle-examples/examples/t5/tokens.json");

    // Open the file in read-only mode
    let mut file = File::open(&path).expect("file not found");

    // Read the file content into a string
    let mut data = String::new();
    file.read_to_string(&mut data).expect("error reading the file");

    // Deserialize the JSON string into a HashMap
    let hashmap: HashMap<String, usize> = serde_json::from_str(&data)?;
    let mut model = builder.build_encoder()?;
    let cnn: &CNN = &builder.build_CNN()?;
    let max_batch = 500; 
    let max_residues = 4000;
    //println!("3");
    /*let tokenizer = tokenizer
        .with_padding(None)
        .with_truncation(None)
        .map_err(E::msg)?;*/


    /*let v = tokenizer.get_vocab(true);
    for key in v.keys() {
        println!("{:?}", key);
    }*/
    //println!("4");
    match args.prompt {
        Some(prompt) => {
            let start = std::time::Instant::now();
            process_fasta(Path::new(&prompt), "C:/Users/SteineggerLab/Desktop/Victor/candle/candle-examples/examples/t5/output.fasta", &mut model, cnn, &hashmap, device);
            //process_fasta_batch(Path::new(&prompt), "C:/Users/SteineggerLab/Desktop/Victor/candle/candle-examples/examples/t5/output.fasta", &mut model, cnn, &hashmap, device, max_residues, max_batch);
            println!("Took {:?}", start.elapsed());
            //let res = predict(prompt, &builder, &hashmap, device)?;
            //println!{"{:?}", res};   
        }
        None => {
            // let mut model = builder.build_encoder()?;
            // let sentences = [
            //     "The cat sits outside",
            //     "A man is playing guitar",
            //     "I love pasta",
            //     "The new movie is awesome",
            //     "The cat plays in the garden",
            //     "A woman watches TV",
            //     "The new movie is so great",
            //     "Do you like pizza?",
            // ];
            // let n_sentences = sentences.len();
            // let mut all_embeddings = Vec::with_capacity(n_sentences);
            // for sentence in sentences {
            //     let tokens = tokenizer.encode(&sentence, None, 1024, &TruncationStrategy::LongestFirst, 0).token_ids;
            //     let token_ids = Tensor::new(&tokens[..], model.device())?.unsqueeze(0)?;
            //     let embeddings = model.forward(&token_ids)?;
            //     println!("generated embeddings {:?}", embeddings.shape());
            //     // Apply some avg-pooling by taking the mean embedding value for all tokens (including padding)
            //     let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
            //     let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;
            //     let embeddings = if args.normalize_embeddings {
            //         normalize_l2(&embeddings)?
            //     } else {
            //         embeddings
            //     };
            //     println!("pooled embeddings {:?}", embeddings.shape());
            //     all_embeddings.push(embeddings)
            // }

            // let mut similarities = vec![];
            // for (i, e_i) in all_embeddings.iter().enumerate() {
            //     for (j, e_j) in all_embeddings
            //         .iter()
            //         .enumerate()
            //         .take(n_sentences)
            //         .skip(i + 1)
            //     {
            //         let sum_ij = (e_i * e_j)?.sum_all()?.to_scalar::<f32>()?;
            //         let sum_i2 = (e_i * e_i)?.sum_all()?.to_scalar::<f32>()?;
            //         let sum_j2 = (e_j * e_j)?.sum_all()?.to_scalar::<f32>()?;
            //         let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
            //         similarities.push((cosine_similarity, i, j))
            //     }
            // }
            // similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
            // for &(score, i, j) in similarities[..5].iter() {
            //     println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
            // }
        }
    }
    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
