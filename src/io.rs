//! Logic to perform filtering through IO

use crate::algos::clean_block;
use byte_slice_cast::{AsMutSliceOf, AsSliceOf};
use color_eyre::eyre::Result;
use faer::prelude::*;
use memmap2::Mmap;
use psrdada::prelude::*;
use sigproc_filterbank::{read::ReadFilterbank, write::WriteFilterbank};
use std::{
    fs::File,
    io::{BufWriter, Write},
};

const CHANNELS: usize = 2048;

//use crate::algos::clean_block;

pub fn clean_filterbank(
    in_file: &str,
    out_file: &str,
    first_pass_sigma: f32,
    second_pass_sigma: f32,
    detrend_order: usize,
) -> Result<()> {
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
    // Dumb cast the byte pointer
    let data_slice = fb_in
        .raw_data
        .as_slice_of()
        .expect("Couldn't reinterpret filterbank slice as f32");

    let data_in: MatRef<'_, f32> =
        mat::from_column_major_slice(data_slice, fb_in.nchans(), fb_in.nsamples());

    // Clone the whole block
    let mut mat = data_in.to_owned();

    // Clean the data
    clean_block(
        mat.as_mut(),
        first_pass_sigma,
        second_pass_sigma,
        detrend_order,
    );

    // Then write each time series to the file
    for i in 0..mat.ncols() {
        let packed = fb_out.pack(mat.col_as_slice(i));
        fb_writer.write_all(&packed)?;
    }
    fb_writer.flush()?;

    Ok(())
}

pub fn clean_psrdada(
    in_key: i32,
    out_key: i32,
    first_pass_sigma: f32,
    second_pass_sigma: f32,
    detrend_order: usize,
) -> Result<()> {
    // Read a block at a time from PSRDADA and write to another PSRDADA buffer
    // Both buffers must exist at runtime, we're not creating them

    // Connect to the two header/data paired buffers
    let mut in_client =
        HduClient::connect(in_key).expect("Could not connect to the input DADA buffer");
    let mut out_client =
        HduClient::connect(out_key).expect("Could not connect to the output DADA buffer");

    // Split these into their header/data pairs
    let (mut in_header, mut in_data) = in_client.split();
    let (mut out_header, mut out_data) = out_client.split();

    // HEIMDALL-Specific, pass along the single header that we get from T0 and continue
    let mut in_header_rdr = in_header
        .reader()
        .expect("Could not lock the input header buffer as a reader");
    let mut out_header_wdr = out_header
        .writer()
        .expect("Could not lock the output header buffer as a writer");

    if let Some(mut in_block) = in_header_rdr.next() {
        if let Some(mut out_block) = out_header_wdr.next() {
            let in_bytes = in_block.block();
            let out_bytes = out_block.block();
            out_bytes.clone_from_slice(in_bytes);
        } else {
            panic!("Couldn't get the next header write block")
        }
    } else {
        panic!("Couldn't get the next header read block")
    }

    // Create the data block readers and writers
    let mut in_data_rdr = in_data
        .reader()
        .expect("Could not lock the input data buffer as a reader");
    let mut out_data_wdr = out_data
        .writer()
        .expect("Could not lock the output data buffer as a writer");

    // Loop forever on reading from the input, applying the transformation and writing to the output
    while let Some(mut read_block) = in_data_rdr.next() {
        // Get the next write block
        if let Some(mut write_block) = out_data_wdr.next() {
            // Get the byte ptrs
            let read_bytes = read_block.block();
            let write_bytes = write_block.block();

            // Start by just copying the data across
            write_bytes.clone_from_slice(read_bytes);

            // First reinterpret the byte slice as a float 32 slice
            let write_floats = write_bytes.as_mut_slice_of()?;
            let samples = write_floats.len() / CHANNELS;

            // Then reinterpret as a matrix
            let mut mat: MatMut<'_, f32> =
                mat::from_column_major_slice_mut(write_floats, CHANNELS, samples);

            // And then do the cleaning
            clean_block(
                mat.as_mut(),
                first_pass_sigma,
                second_pass_sigma,
                detrend_order,
            );

            // Finally, for feeding heimdall, we want to replace every NaN with zero
            for j in 0..samples {
                let mut col = mat.as_mut().col_mut(j);
                zipped!(&mut col).for_each(|unzipped!(mut x)| {
                    if (*x).is_nan() {
                        *x = 0.;
                    }
                })
            }

            // No need to lock, mark cleared, or anything like that. That's all implicit with RAII.
        } else {
            println!("Errored on getting the next write block, perhaps that buffer was destroyed?");
            break;
        }
    }

    Ok(())
}
