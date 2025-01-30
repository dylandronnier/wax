use candle_core::{Result, Tensor};
use candle_datasets::{batcher::IterResult2, Batcher};
use candle_nn::{loss::cross_entropy, ModuleT, Optimizer};
use indicatif::ProgressBar;

pub fn train_loop<M: ModuleT, O: Optimizer, I: Iterator<Item = Result<(Tensor, Tensor)>>>(
    model: &mut M,
    optimizer: &mut O,
    batcher: Batcher<IterResult2<I>>,
) -> Result<()> {
    let pb = ProgressBar::new(3750);
    for batch in batcher {
        let (images, labels) = batch?;
        let images_reshaped = images.reshape(((), 1, 28, 28))?;
        let logits = model.forward_t(&images_reshaped, true)?;
        let loss = cross_entropy(&logits, &labels)?;
        optimizer.backward_step(&loss)?;
        pb.inc(1);
    }
    Ok(())
}

pub fn predict<M: ModuleT>(model: &M, images: &Tensor) -> Result<Tensor> {
    model.forward_t(&images, true)?.argmax(1)
}

pub fn eval_loop<M: ModuleT, I: Iterator<Item = Result<(Tensor, Tensor)>>>(
    model: &M,
    batcher: Batcher<IterResult2<I>>,
) -> Result<(f32, f32)> {
    let mut cumul_loss = 0f32;
    let mut cumul_accuracy = 0f32;
    let pb = ProgressBar::new(625);
    for batch in batcher {
        let (images, labels) = batch?;
        let images_reshaped = images.reshape(((), 1, 28, 28))?;
        let logits = model.forward_t(&images_reshaped, true)?;
        let preds = logits.argmax(1)?.to_dtype(candle_core::DType::U8)?;
        cumul_accuracy +=
            (preds.eq(&labels)?.sum_all()?.to_scalar::<u8>()? as f32) / (preds.elem_count() as f32);
        cumul_loss += cross_entropy(&logits, &labels)?.to_scalar::<f32>()?;
        pb.inc(1);
    }
    Ok((cumul_loss / 625.0, cumul_accuracy / 625.0))
}
