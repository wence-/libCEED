#![doc = include_str!("../README.md")]

/**
Bindings generated from libCEED's C headers using bindgen.

See `build.rs` to customize the process and refer to the [libCEED API
docs](https://libceed.readthedocs.io/en/latest/api/) for usage.
*/
pub mod bind_ceed {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(dead_code)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}
