#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;
use std::path::PathBuf;

#[cfg(feature = "quantize")]
use {
    candle_transformers::models::quantized_t5 as t5,
    candle_transformers::quantized_var_builder::VarBuilder,
};

#[cfg(not(feature = "quantize"))]
use candle_transformers::models::t5;

use anyhow::{Error as E, Result};
use candle::{DType, Device, Tensor, D, IndexOp};
use candle_nn::VarBuilder as BaseVarBuilder;
use candle_nn::{Module, Conv2d, Conv2dConfig,  Activation, Dropout};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

use serde_json; 
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, Write, Read};
use std::path::Path;

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

fn predict(prompt: String, model: &mut t5::T5EncoderModel, cnn: &CNN, hashmap:&HashMap<String,usize>, device: &Device) -> Result<Vec<String>>{
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

    let input_token_ids = Tensor::new(&tokens[..], device)?.unsqueeze(0)?.to_dtype(DType::U8)?;
    
    let ys = model.forward(&input_token_ids)?;
    // FIXME
    let ys = ys.i((.., 1..ys.dims3()?.1-1))?;
    let ys = ys.pad_with_zeros(1, 0, 1)?;
    // let ys = ys.to_dtype(DTYPE)?;
    let output = &cnn.forward(&ys)?;
    let mut structures: Vec<String> = Vec::new();
    let vals = output.argmax_keepdim(1)?;
    let vals = vals.flatten(0, 2)?;
    let vals = vals.to_vec1::<u32>()?;
    let structure: Vec<char> = vals.iter().map(|&n| number_to_char(n)).collect();
    let structure: String = structure.into_iter().collect();
    println!("{:?}", structure);
    structures.push(structure.clone());
    Ok(structures)
}

fn process_fasta(input_path: &Path, output_path: &str, model: &mut t5::T5EncoderModel, cnn: &CNN,  hashmap: &HashMap<String,usize>, device: &Device) -> io::Result<()> {
    let file = File::open(input_path)?;
    let reader = io::BufReader::new(file);
    let mut output_file = OpenOptions::new().create(true).write(true).truncate(true).open(output_path)?;

    let mut s = String::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if !s.is_empty() {
                let prompt = s.clone();
                let res = predict(prompt, model, cnn, hashmap, device);
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
    vb: BaseVarBuilder,
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

pub struct CNN {
    conv1: Conv2d,
    act: Activation,
    dropout: Dropout,
    conv2: Conv2d
}

impl CNN {
    pub fn load(vb: BaseVarBuilder, config: Conv2dConfig) -> Result<Self>{
        let conv1 = conv2d_non_square(1024, 32, 7, 1, config, vb.pp("classifier.0"))?;
        let act = Activation::Relu;
        let dropout = Dropout::new(0.0);
        let conv2 = conv2d_non_square(32, 20, 7, 1, config, vb.pp("classifier.3"))?;
        Ok(Self { conv1, act, dropout, conv2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs: Tensor = xs.permute((0, 2, 1))?.unsqueeze(D::Minus1)?;
        let xs: Tensor = xs.pad_with_zeros(2, 3, 3)?;
        let xs: Tensor = self.conv1.forward(&xs)?;
        let xs: Tensor = xs.relu()?;
        // let xs: Tensor = xs.clone();
        let xs: Tensor = xs.pad_with_zeros(2, 3, 3)?;
        let xs = self.conv2.forward(&xs)?.squeeze(D::Minus1)?;
        Ok(xs)
    }
}

struct T5ModelBuilder {
    device: Device,
    config: t5::Config,
    weights_filename: Vec<PathBuf>,
    cnn_filename: Vec<PathBuf>,
    tokens_filename: Vec<PathBuf>,
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
        let weights_filename = if model_id == "google/flan-t5-xxl" || model_id == "google/flan-ul2"
        {
            candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?
        } else {
            #[cfg(feature = "quantize")]
            {
                vec![api.get("model_q6k.gguf")?]
            }
            #[cfg(not(feature = "quantize"))]
            {
                vec![api.get("model.safetensors")?]
            }
        };
        let cnn_filename = vec![api.get("cnn.safetensors")?];
        let config = std::fs::read_to_string(config_filename)?;
        let mut config: t5::Config = serde_json::from_str(&config)?;
        config.use_cache = !args.disable_cache;

        let tokens_filename = vec![api.get("tokens.json")?];
        Ok(
            Self {
                device,
                config,
                weights_filename,
                cnn_filename,
                tokens_filename,
            }
        )
    }

    pub fn build_encoder(&self) -> Result<t5::T5EncoderModel> {
        #[cfg(feature = "quantize")]
        let vb = t5::VarBuilder::from_gguf(&self.weights_filename[0], &self.device)?;
        #[cfg(not(feature = "quantize"))]
        let vb = unsafe {
            BaseVarBuilder::from_mmaped_safetensors(&self.weights_filename, DTYPE, &self.device)?
        };
        Ok(t5::T5EncoderModel::load(vb, &self.config)?)
    }
    pub fn build_CNN(&self) -> Result<CNN> {
        let vb = unsafe {
            BaseVarBuilder::from_mmaped_safetensors(&self.cnn_filename, DTYPE, &self.device)?
            // VarBuilder::from_gguf(&self.cnn_filename[0], &self.device);
        };
        let config = Conv2dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: 1,
        };
        Ok(CNN::load(vb,config)?)
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

    let builder = T5ModelBuilder::load(&args)?;
    let device = &builder.device;

    // Open the file in read-only mode
    let mut file = File::open(&builder.tokens_filename[0]).expect("file not found");

    // Read the file content into a string
    let mut data = String::new();
    file.read_to_string(&mut data).expect("error reading the file");

    // Deserialize the JSON string into a HashMap
    let hashmap: HashMap<String, usize> = serde_json::from_str(&data)?;
    let mut model = builder.build_encoder()?;
    let cnn: &CNN = &builder.build_CNN()?;
    let max_batch = 500; 
    let max_residues = 4000;
    match args.prompt {
        Some(prompt) => {
            let start = std::time::Instant::now();
            process_fasta(Path::new(&prompt), "output.fasta", &mut model, cnn, &hashmap, device);
            println!("Took {:?}", start.elapsed());
        },
        None => {}
    }
    Ok(())
}
