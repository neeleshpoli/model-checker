mod common;

use common::get_test_model_dir;

use ort_genai::{
    Config,
    Generator,
    GeneratorParams,
    Model,
    Sequences,
    Tensor,
    Tokenizer,
    TokenizerStream,
};


#[test]
fn test_config() {
    crate::assert_no_leak!({

    // We can't run this without linking, but it should compile.
    let model_dir = get_test_model_dir();
    let _config = ort_genai::Config::new(model_dir.to_str().unwrap()).unwrap();
    });
}

#[test]
fn test_model() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        assert!(model.get_type().is_ok());
        assert!(model.get_device_type().is_ok());
    });
}

#[test]
fn test_model_with_config() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let config = Config::new(model_dir.to_str().unwrap()).unwrap();
        let model = Model::new_with_config(&config).unwrap();
        assert!(model.get_type().is_ok());
    });
}

#[test]
fn test_tokenizer() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        let tokenizer = Tokenizer::new(&model).unwrap();

        let mut sequences = Sequences::new().unwrap();

        // Encode
        tokenizer.encode("Hello, world!", &mut sequences).unwrap();

        let mut strings = ["Hello, world!", "Test 2"];
        let _encoded_batch = tokenizer.encode_batch(&mut strings).unwrap();

        // Decode
        let _decoded = tokenizer.decode(sequences.get_sequence_data(0).unwrap()).unwrap();

        let _stream = TokenizerStream::new(&tokenizer).unwrap();
        // Drop stream
    });
}

#[test]
fn test_generator() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();

        let params = GeneratorParams::new(&model).unwrap();

        let mut generator = Generator::new(model, params).unwrap();
        assert!(!generator.is_done());
    });
}

#[test]
fn test_tensor() {
    crate::assert_no_leak!({
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let tensor = Tensor::new_from_buffer(Some(&mut data), &shape).unwrap();
        let _type_ = tensor.get_type().unwrap();
        let shape_out = tensor.get_shape().unwrap();
        assert_eq!(shape_out, shape);
    });
}

#[test]
fn test_generator_params() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();

        let mut params = GeneratorParams::new(&model).unwrap();
        params.set_search_number("max_length", 50.0).unwrap();
        params.set_search_number("top_p", 0.9).unwrap();
    });
}

#[test]
fn test_runtime_settings() {
    crate::assert_no_leak!({
        let _settings = ort_genai::RuntimeSettings::new().unwrap();
    });
}

#[test]
fn test_tokenizer_stream() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        let tokenizer = Tokenizer::new(&model).unwrap();

        let mut stream = TokenizerStream::new(&tokenizer).unwrap();
        let _decoded = stream.decode(42).unwrap();
    });
}

#[test]
fn test_engine() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        // Just verify it compiles, Engine::new takes ownership of Model
        let _engine = ort_genai::Engine::<f32>::new(model);
    });
}

#[test]
fn test_named_tensors() {
    crate::assert_no_leak!({
        let named_tensors = ort_genai::NamedTensors::new().unwrap();
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let mut tensor = Tensor::new_from_buffer(Some(&mut data), &shape).unwrap();
        named_tensors.set_tensor("test_tensor", &mut tensor).unwrap();

        assert_eq!(named_tensors.get_count().unwrap(), 1);
        // let names = named_tensors.get_names().unwrap();
        // assert_eq!(names.len(), 1);
        // assert_eq!(names[0], "test_tensor");

        // let _retrieved_tensor = named_tensors.get_tensor("test_tensor").unwrap();
        named_tensors.delete_tensor("test_tensor").unwrap();
        assert_eq!(named_tensors.get_count().unwrap(), 0);
    });
}

#[test]
fn test_multi_modal_processor() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        // Since Phi-3-mini is text only this might fail at runtime or load, but we test compilation
        // and if it doesn't panic on drop.
        let _processor = ort_genai::MultiModalProcessor::new(&model);
    });
}

#[test]
fn test_streaming_processor() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        // Just verify compilation since it requires a nemotron_speech model.
        let _processor = ort_genai::StreamingProcessor::new(model);
    });
}

#[test]
fn test_request_opaque() {
    let model_dir = get_test_model_dir();
    crate::assert_no_leak!({
        let mut data = 42;
        let model = Model::new(model_dir.to_str().unwrap()).unwrap();
        let mut params = GeneratorParams::new(&model).unwrap();

        let mut request = ort_genai::Request::<i32>::new(&mut params).unwrap();

        request.set_opaque_data(&mut data).unwrap();
        let retrieved_data = request.get_opaque_data().unwrap();
        assert_eq!(*retrieved_data, 42);
    });
}

#[test]
fn test_element_type() {
    let t: ort_genai::ElementType = ort_genai_sys::OgaElementType_OgaElementType_float32.into();
    assert_eq!(t, ort_genai::ElementType::Float32);
    let oga_t: ort_genai_sys::OgaElementType = ort_genai::ElementType::Float32.into();
    assert_eq!(oga_t, ort_genai_sys::OgaElementType_OgaElementType_float32);
}

#[test]
fn test_error() {
    let err = ort_genai::Error::NulError;
    assert_eq!(err.to_string(), "Null found in parameter!");
}

#[test]
fn test_lib_globals() {
    ort_genai::telemetry_enabled(false);

    // We can't really register/unregister without a real library, but we can call basic ones safely.
    let _ = ort_genai::get_current_gpu_device_id();

    // shutdown() shouldn't be called if other tests are running
}
#[test]
fn test_string_array() {
}
