use ort_genai::{StringArray};

#[test]
fn test_string_array_repeated_get() {
    let strings = ["hello", "world"];
    let array = StringArray::from_strings(&strings).unwrap();
    assert_eq!(array.len().unwrap(), 2);

    // Test repeated get (should not corrupt memory or double free)
    for _ in 0..5 {
        assert_eq!(array.get(0).unwrap(), "hello");
        assert_eq!(array.get(1).unwrap(), "world");
    }
}

// As tests are prevented from linking properly in the sandbox due to missing `.so` dependencies on linux container compared to `.dll`,
// these integration tests provide compilation validations of our public API without full linking at runtime currently.
