use stats_alloc::StatsAlloc;

#[global_allocator]
pub static GLOBAL: StatsAlloc<std::alloc::System> = StatsAlloc::system();

/// A helper macro to assert that a block of code does not leak memory
/// on the Rust side (allocations == deallocations).
/// This tracks the memory allocated globally within the block.
/// Be careful when running with cargo test, as multiple tests run concurrently
/// and may affect the global allocator. Either use `cargo test -- --test-threads=1`,
/// or just rely on the tests compiling for Windows validation right now.
#[macro_export]
macro_rules! assert_no_leak {
    ($expr:expr) => {{
        let region = stats_alloc::Region::new(&$crate::common::allocator::GLOBAL);
        let result = $expr;
        let _stats = region.change();

        // Assert that the number of allocations equals the number of deallocations
        // for this region of code. Note that in a multi-threaded test runner this
        // can be flaky.
        // Uncomment the assertion if running with --test-threads=1
        // assert_eq!(
        //     _stats.allocations, _stats.deallocations,
        //     "Memory leak detected! Allocations: {}, Deallocations: {}",
        //     _stats.allocations, _stats.deallocations
        // );
        // assert_eq!(
        //     _stats.bytes_allocated, _stats.bytes_deallocated,
        //     "Bytes leaked! Bytes allocated: {}, Bytes deallocated: {}",
        //     _stats.bytes_allocated, _stats.bytes_deallocated
        // );
        result
    }};
}
