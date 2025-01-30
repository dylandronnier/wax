use candle_core::Result;
use candle_nn::{
    batch_norm, conv2d, linear, BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, Module,
    ModuleT, VarBuilder,
};

struct BasicBlock {
    convolution_layer: Conv2d,
    batch_norm: BatchNorm,
}

impl ModuleT for BasicBlock {
    fn forward_t(&self, xs: &candle_core::Tensor, train: bool) -> Result<candle_core::Tensor> {
        let xs = self.convolution_layer.forward(xs)?;
        self.batch_norm.forward_t(&xs, train)?.relu()
    }
}

fn basic_block(
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    vb: VarBuilder,
) -> Result<BasicBlock> {
    let convolution_layer = conv2d(
        in_channels,
        out_channels,
        kernel_size,
        Conv2dConfig {
            padding: 1,
            // stride: 0,
            ..Default::default()
        },
        vb.pp("convolution_layer"),
    )?;
    let batch_norm = batch_norm(
        out_channels,
        BatchNormConfig {
            momentum: 0.9,
            ..Default::default()
        },
        vb.pp("batch_norm"),
    )?;
    Ok(BasicBlock {
        convolution_layer,
        batch_norm,
    })
}

struct SimpleCNNBlock {
    cnn_layers: Vec<BasicBlock>,
}

fn simple_cnn_block(
    in_channels: usize,
    out_channels: usize,
    nb_conv_layers: usize,
    vb: VarBuilder,
) -> Result<SimpleCNNBlock> {
    let mut cnn_layers = Vec::with_capacity(nb_conv_layers);
    let bb = basic_block(in_channels, out_channels, 3, vb.pp("pre_layer"))?;
    cnn_layers.push(bb);
    for i in 1..(nb_conv_layers + 1) {
        let bb = basic_block(out_channels, out_channels, 3, vb.pp(format!("block_{i}")))?;
        cnn_layers.push(bb);
    }
    // let second_block = basic_block(out_channels, out_channels, 3, vb.pp("second_block"))?;
    Ok(SimpleCNNBlock { cnn_layers })
}

impl ModuleT for SimpleCNNBlock {
    fn forward_t(
        &self,
        xs: &candle_core::Tensor,
        train: bool,
    ) -> candle_core::Result<candle_core::Tensor> {
        let mut xs = xs.clone();
        for b in self.cnn_layers.iter() {
            xs = b.forward_t(&xs, train)?;
        }
        xs = xs.max_pool2d_with_stride(2, 2)?;
        Ok(xs)
    }
}

pub struct SimpleCNN {
    cnn_part: Vec<SimpleCNNBlock>,
    head: Vec<Linear>,
}

pub fn simplecnn(vb: VarBuilder) -> Result<SimpleCNN> {
    let mut cnn_part = Vec::with_capacity(2);
    let mut head = Vec::with_capacity(2);
    let bb = simple_cnn_block(1, 32, 2, vb.pp("first_block"))?;
    cnn_part.push(bb);
    let bb = simple_cnn_block(32, 64, 2, vb.pp("second_block"))?;
    cnn_part.push(bb);
    let bb = linear(3136, 256, vb.pp("fully_connected"))?;
    head.push(bb);
    let bb = linear(256, 10, vb.pp("head"))?;
    head.push(bb);
    Ok(SimpleCNN { cnn_part, head })
}

impl ModuleT for SimpleCNN {
    fn forward_t(&self, xs: &candle_core::Tensor, train: bool) -> Result<candle_core::Tensor> {
        let mut xs = xs.clone();
        for b in self.cnn_part.iter() {
            xs = b.forward_t(&xs, train)?;
        }
        xs = xs.flatten_from(1)?;
        for b in self.head.iter() {
            xs = b.forward_t(&xs, train)?;
        }
        Ok(xs)
    }
}
