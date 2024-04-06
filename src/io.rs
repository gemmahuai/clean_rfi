//! Logic to perform filtering through IO

use color_eyre::eyre::Result;
use faer::prelude::*;
use memmap2::Mmap;
use sigproc_filterbank::{read::ReadFilterbank, write::WriteFilterbank};
use std::{
    fs::File,
    io::{BufWriter, Write},
};

use crate::algos::clean_block;

pub fn clean_filterbank(in_file: &str, out_file: &str) -> Result<()> {
    // Open and parse the input
    let fb_in_file = File::open(in_file)?;
    let fb_in_mm = unsafe { Mmap::map(&fb_in_file)? };
    let fb_in = ReadFilterbank::from_bytes(&fb_in_mm[..])?;

    // Create the output filterbank object
    let mut fb_out = WriteFilterbank::new(fb_in.nchans(), fb_in.nifs());

    // Copy the headers
    fb_out.telescope_id = fb_in.telescope_id();
    fb_out.machine_id = fb_in.machine_id();
    fb_out.data_type = fb_in.data_type();
    fb_out.source_name = fb_in.source_name().map(|x| x.to_owned());
    fb_out.barycentric = fb_in.barycentric();
    fb_out.pulsarcentric = fb_in.pulsarcentric();
    fb_out.az_start = fb_in.az_start();
    fb_out.za_start = fb_in.za_start();
    fb_out.src_raj = fb_in.src_raj();
    fb_out.src_dej = fb_in.src_dej();
    fb_out.tstart = fb_in.tstart();
    fb_out.tsamp = fb_in.tsamp();
    fb_out.fch1 = fb_in.fch1();
    fb_out.foff = fb_in.foff();
    fb_out.ref_dm = fb_in.ref_dm();
    fb_out.period = fb_in.period();
    fb_out.nbeams = fb_in.nbeams();
    fb_out.ibeam = fb_in.ibeam();

    // Set the rawdatafile header, as that's technically what it's here for
    fb_out.rawdatafile = Some(in_file.to_owned());

    // Create the output file
    let fb_out_file = File::create(out_file).expect("Could not create file");
    let mut fb_writer = BufWriter::new(fb_out_file);

    // Write the header
    fb_writer.write_all(&fb_out.header_bytes()).unwrap();
    fb_writer.flush()?;

    // Copy the data into an in-memory buffer TODO: Make this fast and not suck
    // NOTE: Faer is column-major, so time will be in cols and freq in rows such that subsequent freqeuncy channels are memory-contiguous
    let mut data = Mat::from_fn(fb_in.nchans(), fb_in.nsamples(), |i, j| fb_in.get(0, j, i));

    // Clean the data
    clean_block(data.as_mut());

    // Then write each time series to the file
    for i in 0..data.ncols() {
        let packed = fb_out.pack(data.col_as_slice(i));
        fb_writer.write_all(&packed)?;
    }
    fb_writer.flush()?;

    Ok(())
}

pub fn clean_psrdada() {
    // Read a block at a time from PSRDADA and write to another PSRDADA buffer
    // Both buffers must exist at runtime, we're not creating them
    todo!()
}
