use clap::Parser;
use std::{env, path::PathBuf};

use candle_core::{Device, Result};
use candle_datasets::{
    vision::{mnist, Dataset},
    Batcher,
};
use candle_nn::{Optimizer, VarBuilder, VarMap, SGD};
use dataiter::DataIter;
use model::simplecnn;
use training::{eval_loop, train_loop};

mod dataiter;
mod model;
mod training;

fn get_default_model_path() -> PathBuf {
    let mut path = env::current_exe().unwrap();
    path.pop();
    path.push("cnn.safetensors");
    path
}

/// Train a Convolutional Neural Network on the MNIST dataset and save the weights.
#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Cli {
    // File path where to save the trained model
    #[arg(short, long, default_value = get_default_model_path().into_os_string())]
    model_path: PathBuf,
}

fn train(dataset: &Dataset, saving_path: PathBuf, device: &Device) -> Result<()> {
    let var = VarMap::new();
    let varb = VarBuilder::from_varmap(&var, candle_core::DType::F32, device);
    let mut simple = simplecnn(varb)?;
    let mut optimizer = SGD::new(var.all_vars(), 0.002).unwrap();
    let dataloader = DataIter::new_random(&dataset.train_images, &dataset.train_labels)?;
    let batcher = Batcher::new_r2(dataloader);
    train_loop(&mut simple, &mut optimizer, batcher)?;
    let dataloader = DataIter::new(&dataset.test_images, &dataset.test_labels)?;
    let batcher = Batcher::new_r2(dataloader);
    let (loss, accuracy) = eval_loop(&simple, batcher).unwrap();
    println!("Test loss: {}, Test accuracy: {}", loss, accuracy);
    var.save(saving_path)?;
    Ok(())
}

fn main() {
    let args = Cli::parse();
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    let mnist = mnist::load().unwrap();
    train(&mnist, args.model_path, &device).unwrap();
}
