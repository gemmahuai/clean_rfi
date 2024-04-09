use clap::{Parser, Subcommand};
use clean_rfi::io::{clean_filterbank, clean_psrdada};
use color_eyre::eyre::Result;

#[derive(Parser, Debug)]
struct Args {
    #[command(subcommand)]
    method: Method,
    /// Polynomial order used in detrending
    #[clap(short, default_value_t = 4)]
    detrend_order: usize,
    /// Standard deviation threshold in first pass
    #[clap(short, default_value_t = 3.0)]
    first_pass_sigma: f32,
    /// Standard deviation threshold in second pass
    #[clap(short, default_value_t = 5.0)]
    second_pass_sigma: f32,
}

#[derive(Subcommand, Debug)]
enum Method {
    /// Clean RFI between a stream of PSRDADA buffers
    Dada {
        #[clap(short, value_parser = valid_dada_key)]
        /// Hex key of DADA buffer to read from
        from: i32,
        #[clap(short, value_parser = valid_dada_key)]
        /// Hex key of DADA buffer to write to
        to: i32,
    },
    /// Clean RFI from a filterbank file
    Filterbank {
        #[clap(short)]
        /// Filterbank file to read from
        from: String,
        #[clap(short)]
        /// Filterbank file to write to
        to: String,
    },
}

fn valid_dada_key(s: &str) -> Result<i32, String> {
    i32::from_str_radix(s, 16).map_err(|_| "Invalid hex litteral".to_string())
}

fn main() -> Result<()> {
    let args = Args::parse();
    color_eyre::install()?;

    match args.method {
        Method::Dada { from, to } => clean_psrdada(
            from,
            to,
            args.first_pass_sigma,
            args.second_pass_sigma,
            args.detrend_order,
        )?,
        Method::Filterbank { from, to } => clean_filterbank(
            &from,
            &to,
            args.first_pass_sigma,
            args.second_pass_sigma,
            args.detrend_order,
        )?,
    };

    Ok(())
}
