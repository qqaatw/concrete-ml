//! Compiler module

use crate::mlir::ffi::*;
use std::os::raw::c_char;
use std::{ffi::CStr, path::Path};

#[derive(Debug)]
pub struct CompilerError(String);

/// Retreive buffer of the error message from a C struct.
trait CStructErrorMsg {
    fn error_msg(&self) -> *const i8;
}

/// All C struct can return a pointer to the allocated error message.
macro_rules! impl_CStructErrorMsg {
    ([$($t:ty),+]) => {
        $(impl CStructErrorMsg for $t {
            fn error_msg(&self) -> *const i8 {
                self.error
            }
        })*
    }
}
impl_CStructErrorMsg! {[
    crate::mlir::ffi::LibrarySupport,
    CompilationResult,
    LibraryCompilationResult,
    ServerLambda,
    ClientParameters,
    PublicArguments,
    PublicResult,
    KeySet,
    KeySetCache,
    LambdaArgument,
    BufferRef,
    EvaluationKeys
]}

/// Construct a rust error message from a buffer in the C struct.
fn get_error_msg_from_ctype<T: CStructErrorMsg>(c_struct: T) -> String {
    unsafe {
        let error_msg_cstr = CStr::from_ptr(c_struct.error_msg());
        String::from(error_msg_cstr.to_str().unwrap())
    }
}

/// Create string from an MlirStringRef and free its memory.
///
/// # SAFETY
///
/// This should only be used with string refs returned by the compiler.
unsafe fn mlir_string_ref_to_string(str_ref: MlirStringRef) -> String {
    let result = String::from_utf8_lossy(std::slice::from_raw_parts(
        str_ref.data as *const u8,
        str_ref.length as usize,
    ))
    .to_string();
    mlirStringRefDestroy(str_ref);
    result
}

/// Create a vector of bytes from a BufferRef and free its memory.
///
/// # SAFETY
///
/// This should only be used with string refs returned by the compiler.
unsafe fn buffer_ref_to_bytes(buffer_ref: BufferRef) -> Vec<c_char> {
    let result =
        std::slice::from_raw_parts(buffer_ref.data as *const c_char, buffer_ref.length as usize)
            .to_vec();
    bufferRefDestroy(buffer_ref);
    result
}

/// Parse the MLIR code and returns it.
///
/// The function parse the provided MLIR textual representation and returns it. It would fail with
/// an error message to stderr reporting what's bad with the parsed IR.
///
/// # Examples
/// ```
/// use concrete_compiler_rust::compiler::*;
///
/// let module_to_compile = "
///     func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
///         %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
///         return %0 : !FHE.eint<5>
///     }";
/// let result_str = round_trip(module_to_compile);
/// ```
///
pub fn round_trip(mlir_code: &str) -> Result<String, CompilerError> {
    unsafe {
        let engine = compilerEngineCreate();
        let mlir_code_buffer = mlir_code.as_bytes();
        let compilation_result = compilerEngineCompile(
            engine,
            MlirStringRef {
                data: mlir_code_buffer.as_ptr() as *const c_char,
                length: mlir_code_buffer.len() as size_t,
            },
            CompilationTarget_ROUND_TRIP,
        );
        if compilationResultIsNull(compilation_result) {
            let error_msg = get_error_msg_from_ctype(compilation_result);
            compilationResultDestroy(compilation_result);
            return Err(CompilerError(format!(
                "Error in compiler (check logs for more info): {}",
                error_msg
            )));
        }
        let module_compiled = compilationResultGetModuleString(compilation_result);
        let result_str = mlir_string_ref_to_string(module_compiled);
        compilerEngineDestroy(engine);
        Ok(result_str)
    }
}

/// Support for compiling and executing libraries.
pub struct LibrarySupport {
    support: crate::mlir::ffi::LibrarySupport,
}

impl Drop for LibrarySupport {
    fn drop(&mut self) {
        unsafe {
            librarySupportDestroy(self.support);
        }
    }
}

impl LibrarySupport {
    /// LibrarySupport manages build files generated by the compiler under the `output_dir_path`.
    ///
    /// The compiled library needs to link to the runtime for proper execution.
    pub fn new(
        output_dir_path: &str,
        runtime_library_path: &str,
    ) -> Result<LibrarySupport, CompilerError> {
        unsafe {
            let output_dir_path_buffer = output_dir_path.as_bytes();
            let runtime_library_path_buffer = runtime_library_path.as_bytes();
            let support = librarySupportCreateDefault(
                MlirStringRef {
                    data: output_dir_path_buffer.as_ptr() as *const c_char,
                    length: output_dir_path_buffer.len() as size_t,
                },
                MlirStringRef {
                    data: runtime_library_path_buffer.as_ptr() as *const c_char,
                    length: runtime_library_path_buffer.len() as size_t,
                },
            );
            if librarySupportIsNull(support) {
                let error_msg = get_error_msg_from_ctype(support);
                librarySupportDestroy(support);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(LibrarySupport { support })
        }
    }

    /// Compile an MLIR into a library.
    pub fn compile(
        &self,
        mlir_code: &str,
        options: Option<CompilationOptions>,
    ) -> Result<LibraryCompilationResult, CompilerError> {
        unsafe {
            let options = options.unwrap_or_else(|| compilationOptionsCreateDefault());
            let mlir_code_buffer = mlir_code.as_bytes();
            let result = librarySupportCompile(
                self.support,
                MlirStringRef {
                    data: mlir_code_buffer.as_ptr() as *const c_char,
                    length: mlir_code_buffer.len() as size_t,
                },
                options,
            );
            if libraryCompilationResultIsNull(result) {
                let error_msg = get_error_msg_from_ctype(result);
                libraryCompilationResultDestroy(result);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(result)
        }
    }

    /// Load server lambda from a compilation result.
    ///
    /// This can be used for executing the compiled function.
    pub fn load_server_lambda(
        &self,
        result: LibraryCompilationResult,
    ) -> Result<ServerLambda, CompilerError> {
        unsafe {
            let server = librarySupportLoadServerLambda(self.support, result);
            if serverLambdaIsNull(server) {
                let error_msg = get_error_msg_from_ctype(server);
                serverLambdaDestroy(server);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(server)
        }
    }

    /// Load client parameters from a compilation result.
    ///
    /// This can be used for creating keys for the compiled library.
    pub fn load_client_parameters(
        &self,
        result: LibraryCompilationResult,
    ) -> Result<ClientParameters, CompilerError> {
        unsafe {
            let params = librarySupportLoadClientParameters(self.support, result);
            if clientParametersIsNull(params) {
                let error_msg = get_error_msg_from_ctype(params);
                clientParametersDestroy(params);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(params)
        }
    }

    /// Run a compiled circuit.
    pub fn server_lambda_call(
        &self,
        server_lambda: ServerLambda,
        args: PublicArguments,
        eval_keys: EvaluationKeys,
    ) -> Result<PublicResult, CompilerError> {
        unsafe {
            let result = librarySupportServerCall(self.support, server_lambda, args, eval_keys);
            if publicResultIsNull(result) {
                let error_msg = get_error_msg_from_ctype(result);
                publicResultDestroy(result);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(result)
        }
    }

    /// Get path to the compiled shared library
    pub fn get_shared_lib_path(&self) -> String {
        unsafe { mlir_string_ref_to_string(librarySupportGetSharedLibPath(self.support)) }
    }

    /// Get path to the client parameters
    pub fn get_client_parameters_path(&self) -> String {
        unsafe { mlir_string_ref_to_string(librarySupportGetClientParametersPath(self.support)) }
    }
}

/// Support for keygen, encryption, and decryption.
///
/// Manages cache for keys if provided during creation.
pub struct ClientSupport {
    client_params: crate::mlir::ffi::ClientParameters,
    key_set_cache: Option<KeySetCache>,
}

impl Drop for ClientSupport {
    fn drop(&mut self) {
        unsafe {
            clientParametersDestroy(self.client_params);
            match self.key_set_cache {
                Some(cache) => keySetCacheDestroy(cache),
                None => (),
            }
        }
    }
}

impl ClientSupport {
    pub fn new(
        client_params: ClientParameters,
        key_set_cache_path: Option<&Path>,
    ) -> Result<ClientSupport, CompilerError> {
        unsafe {
            let key_set_cache = match key_set_cache_path {
                Some(path) => {
                    let cache_path_buffer = path.to_str().unwrap().as_bytes();
                    let cache = keySetCacheCreate(MlirStringRef {
                        data: cache_path_buffer.as_ptr() as *const c_char,
                        length: cache_path_buffer.len() as size_t,
                    });
                    if keySetCacheIsNull(cache) {
                        let error_msg = get_error_msg_from_ctype(cache);
                        keySetCacheDestroy(cache);
                        return Err(CompilerError(format!(
                            "Error in compiler (check logs for more info): {}",
                            error_msg
                        )));
                    }
                    Some(cache)
                }
                None => None,
            };
            Ok(ClientSupport {
                client_params,
                key_set_cache,
            })
        }
    }

    /// Fetch a keyset based on the client parameters, and the different seeds.
    ///
    /// If a cache has already been set, this operation would first try to load an existing key,
    /// and generate a new one if no compatible keyset exists.
    pub fn keyset(
        &self,
        seed_msb: Option<u64>,
        seed_lsb: Option<u64>,
    ) -> Result<KeySet, CompilerError> {
        unsafe {
            let key_set = match self.key_set_cache {
                Some(cache) => keySetCacheLoadOrGenerateKeySet(
                    cache,
                    self.client_params,
                    seed_msb.unwrap_or(0),
                    seed_lsb.unwrap_or(0),
                ),
                None => keySetGenerate(
                    self.client_params,
                    seed_msb.unwrap_or(0),
                    seed_lsb.unwrap_or(0),
                ),
            };
            if keySetIsNull(key_set) {
                let error_msg = get_error_msg_from_ctype(key_set);
                keySetDestroy(key_set);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(key_set)
        }
    }

    /// Encrypt arguments of a compiled circuit.
    pub fn encrypt_args(
        &self,
        args: &[LambdaArgument],
        key_set: KeySet,
    ) -> Result<PublicArguments, CompilerError> {
        unsafe {
            let public_args = lambdaArgumentEncrypt(
                args.as_ptr(),
                args.len() as u64,
                self.client_params,
                key_set,
            );
            if publicArgumentsIsNull(public_args) {
                let error_msg = get_error_msg_from_ctype(public_args);
                publicArgumentsDestroy(public_args);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(public_args)
        }
    }

    pub fn decrypt_result(
        &self,
        result: PublicResult,
        key_set: KeySet,
    ) -> Result<LambdaArgument, CompilerError> {
        unsafe {
            let arg = publicResultDecrypt(result, key_set);
            if lambdaArgumentIsNull(arg) {
                let error_msg = get_error_msg_from_ctype(arg);
                lambdaArgumentDestroy(arg);
                return Err(CompilerError(format!(
                    "Error in compiler (check logs for more info): {}",
                    error_msg
                )));
            }
            Ok(arg)
        }
    }
}

// TODO: implement traits for C Struct that are serializable and reduce code for serialization and maybe refactor other functions.
// destroy and is_null could be implemented for other struct as well.
//
// trait Serializable {
//     fn into_buffer_ref(self) -> BufferRef;
//     fn from_buffer_ref(buff: BufferRef, params: Option<ClientParameters>) -> Self;
//     fn is_null(self) -> bool;
//     fn destroy(self);
// }

// fn serialize<T: Serializable>(to_serialize: T) -> Result<Vec<c_char>, CompilerError> {
//     unsafe {
//         let serialized_ref = to_serialize.into_buffer_ref();
//         if bufferRefIsNull(serialized_ref) {
//             let error_msg = get_error_msg_from_ctype(serialized_ref);
//             bufferRefDestroy(serialized_ref);
//             return Err(CompilerError(error_msg));
//         }
//         let serialized = buffer_ref_to_bytes(serialized_ref);
//         Ok(serialized)
//     }
// }

// fn unserialize<T: Serializable + CStructErrorMsg>(
//     serialized: &Vec<c_char>,
//     client_parameters: Option<ClientParameters>,
// ) -> Result<T, CompilerError> {
//     unsafe {
//         let serialized_ref = bufferRefCreate(
//             serialized.as_ptr() as *const c_char,
//             serialized.len().try_into().unwrap(),
//         );
//         let serialized = T::from_buffer_ref(serialized_ref, client_parameters);
//         if serialized.is_null() {
//             let error_msg = get_error_msg_from_ctype(serialized);
//             serialized.destroy();
//             return Err(CompilerError(error_msg));
//         }
//         Ok(serialized)
//     }
// }

impl PublicArguments {
    pub fn serialize(self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = publicArgumentsSerialize(self);
            if bufferRefIsNull(serialized_ref) {
                let error_msg = get_error_msg_from_ctype(serialized_ref);
                bufferRefDestroy(serialized_ref);
                return Err(CompilerError(error_msg));
            }
            let serialized = buffer_ref_to_bytes(serialized_ref);
            Ok(serialized)
        }
    }
    pub fn unserialize(
        serialized: &Vec<c_char>,
        client_parameters: ClientParameters,
    ) -> Result<PublicArguments, CompilerError> {
        unsafe {
            let serialized_ref = bufferRefCreate(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            );
            let public_args = publicArgumentsUnserialize(serialized_ref, client_parameters);
            if publicArgumentsIsNull(public_args) {
                let error_msg = get_error_msg_from_ctype(public_args);
                publicArgumentsDestroy(public_args);
                return Err(CompilerError(error_msg));
            }
            Ok(public_args)
        }
    }
}

impl PublicResult {
    pub fn serialize(self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = publicResultSerialize(self);
            if bufferRefIsNull(serialized_ref) {
                let error_msg = get_error_msg_from_ctype(serialized_ref);
                bufferRefDestroy(serialized_ref);
                return Err(CompilerError(error_msg));
            }
            let serialized = buffer_ref_to_bytes(serialized_ref);
            Ok(serialized)
        }
    }
    pub fn unserialize(
        serialized: &Vec<c_char>,
        client_parameters: ClientParameters,
    ) -> Result<PublicResult, CompilerError> {
        unsafe {
            let serialized_ref = bufferRefCreate(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            );
            let public_result = publicResultUnserialize(serialized_ref, client_parameters);
            if publicResultIsNull(public_result) {
                let error_msg = get_error_msg_from_ctype(public_result);
                publicResultDestroy(public_result);
                return Err(CompilerError(error_msg));
            }
            Ok(public_result)
        }
    }
}

impl EvaluationKeys {
    pub fn serialize(self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = evaluationKeysSerialize(self);
            if bufferRefIsNull(serialized_ref) {
                let error_msg = get_error_msg_from_ctype(serialized_ref);
                bufferRefDestroy(serialized_ref);
                return Err(CompilerError(error_msg));
            }
            let serialized = buffer_ref_to_bytes(serialized_ref);
            Ok(serialized)
        }
    }
    pub fn unserialize(serialized: &Vec<c_char>) -> Result<EvaluationKeys, CompilerError> {
        unsafe {
            let serialized_ref = bufferRefCreate(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            );
            let eval_keys = evaluationKeysUnserialize(serialized_ref);
            if evaluationKeysIsNull(eval_keys) {
                let error_msg = get_error_msg_from_ctype(eval_keys);
                evaluationKeysDestroy(eval_keys);
                return Err(CompilerError(error_msg));
            }
            Ok(eval_keys)
        }
    }
}

impl ClientParameters {
    pub fn serialize(self) -> Result<Vec<c_char>, CompilerError> {
        unsafe {
            let serialized_ref = clientParametersSerialize(self);
            if bufferRefIsNull(serialized_ref) {
                let error_msg = get_error_msg_from_ctype(serialized_ref);
                bufferRefDestroy(serialized_ref);
                return Err(CompilerError(error_msg));
            }
            let serialized = buffer_ref_to_bytes(serialized_ref);
            Ok(serialized)
        }
    }
    pub fn unserialize(serialized: &Vec<c_char>) -> Result<ClientParameters, CompilerError> {
        unsafe {
            let serialized_ref = bufferRefCreate(
                serialized.as_ptr() as *const c_char,
                serialized.len().try_into().unwrap(),
            );
            let params = clientParametersUnserialize(serialized_ref);
            if clientParametersIsNull(params) {
                let error_msg = get_error_msg_from_ctype(params);
                clientParametersDestroy(params);
                return Err(CompilerError(error_msg));
            }
            Ok(params)
        }
    }
}

#[cfg(test)]
mod test {
    use std::env;
    use tempdir::TempDir;

    use super::*;

    #[test]
    fn test_compiler_round_trip() {
        let module_to_compile = "
                func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let result_str = round_trip(module_to_compile).unwrap();
        let expected_module = "module {
  func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
    return %0 : !FHE.eint<5>
  }
}
";
        assert_eq!(expected_module, result_str);
    }

    #[test]
    fn test_compiler_round_trip_invalid_mlir() {
        let module_to_compile = "bla bla bla";
        let result_str = round_trip(module_to_compile);
        assert!(
            matches!(result_str, Err(CompilerError(err)) if err == "Error in compiler (check logs for more info): Could not parse source\n")
        );
    }

    #[test]
    fn test_compiler_compile_lib() {
        unsafe {
            let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
            let runtime_library_path = match env::var("CONCRETE_COMPILER_BUILD_DIR") {
                Ok(val) => val + "/lib/libConcretelangRuntime.so",
                Err(_e) => "".to_string(),
            };
            let temp_dir = TempDir::new("rust_test_compiler_compile_lib").unwrap();
            let support = LibrarySupport::new(
                temp_dir.path().to_str().unwrap(),
                runtime_library_path.as_str(),
            )
            .unwrap();
            let lib = support.compile(module_to_compile, None).unwrap();
            assert!(!libraryCompilationResultIsNull(lib));
            libraryCompilationResultDestroy(lib);
            // the sharedlib should be enough as a sign that the compilation worked
            assert!(Path::new(support.get_shared_lib_path().as_str()).exists());
            assert!(Path::new(support.get_client_parameters_path().as_str()).exists());
        }
    }

    /// We want to make sure setting a pointer to null in rust passes the nullptr check in C/Cpp
    #[test]
    fn test_compiler_null_ptr_compatibility() {
        unsafe {
            let lib = Library {
                ptr: std::ptr::null_mut(),
                error: std::ptr::null_mut(),
            };
            assert!(libraryIsNull(lib));
        }
    }

    #[test]
    fn test_compiler_load_server_lambda_and_client_parameters() {
        unsafe {
            let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
            let runtime_library_path = match env::var("CONCRETE_COMPILER_BUILD_DIR") {
                Ok(val) => val + "/lib/libConcretelangRuntime.so",
                Err(_e) => "".to_string(),
            };
            let temp_dir = TempDir::new("rust_test_compiler_load_server_lambda").unwrap();
            let support = LibrarySupport::new(
                temp_dir.path().to_str().unwrap(),
                runtime_library_path.as_str(),
            )
            .unwrap();
            let result = support.compile(module_to_compile, None).unwrap();
            let server = support.load_server_lambda(result).unwrap();
            assert!(!serverLambdaIsNull(server));
            serverLambdaDestroy(server);
            let client_params = support.load_client_parameters(result).unwrap();
            assert!(!clientParametersIsNull(client_params));
            libraryCompilationResultDestroy(result);
        }
    }

    #[test]
    fn test_compiler_compile_and_exec_scalar_args() {
        unsafe {
            let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
            let runtime_library_path = match env::var("CONCRETE_COMPILER_BUILD_DIR") {
                Ok(val) => val + "/lib/libConcretelangRuntime.so",
                Err(_e) => "".to_string(),
            };
            let temp_dir = TempDir::new("rust_test_compiler_compile_and_exec_scalar_args").unwrap();
            let lib_support = LibrarySupport::new(
                temp_dir.path().to_str().unwrap(),
                runtime_library_path.as_str(),
            )
            .unwrap();
            // compile
            let result = lib_support.compile(module_to_compile, None).unwrap();
            // loading materials from compilation
            // - server_lambda: used for execution
            // - client_parameters: used for keygen, encryption, and evaluation keys
            let server_lambda = lib_support.load_server_lambda(result).unwrap();
            let client_params = lib_support.load_client_parameters(result).unwrap();
            let client_support = ClientSupport::new(client_params, None).unwrap();
            let key_set = client_support.keyset(None, None).unwrap();
            let eval_keys = keySetGetEvaluationKeys(key_set);
            // build lambda arguments from scalar and encrypt them
            let args = [lambdaArgumentFromScalar(4), lambdaArgumentFromScalar(2)];
            let encrypted_args = client_support.encrypt_args(&args, key_set).unwrap();
            // free args
            args.map(|arg| lambdaArgumentDestroy(arg));
            // execute the compiled function on the encrypted arguments
            let encrypted_result = lib_support
                .server_lambda_call(server_lambda, encrypted_args, eval_keys)
                .unwrap();
            // decrypt the result of execution
            let result_arg = client_support
                .decrypt_result(encrypted_result, key_set)
                .unwrap();
            // get the scalar value from the result lambda argument
            let result = lambdaArgumentGetScalar(result_arg);
            assert_eq!(result, 6);
        }
    }

    #[test]
    fn test_compiler_compile_and_exec_with_serialization() {
        unsafe {
            let module_to_compile = "
            func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
            let runtime_library_path = match env::var("CONCRETE_COMPILER_BUILD_DIR") {
                Ok(val) => val + "/lib/libConcretelangRuntime.so",
                Err(_e) => "".to_string(),
            };
            let temp_dir =
                TempDir::new("rust_test_compiler_compile_and_exec_with_serialization").unwrap();
            let lib_support = LibrarySupport::new(
                temp_dir.path().to_str().unwrap(),
                runtime_library_path.as_str(),
            )
            .unwrap();
            // compile
            let result = lib_support.compile(module_to_compile, None).unwrap();
            // loading materials from compilation
            // - server_lambda: used for execution
            // - client_parameters: used for keygen, encryption, and evaluation keys
            let server_lambda = lib_support.load_server_lambda(result).unwrap();
            let client_params = lib_support.load_client_parameters(result).unwrap();
            // serialize client parameters
            let serialized_params = client_params.serialize().unwrap();
            let client_params = ClientParameters::unserialize(&serialized_params).unwrap();
            // create client support
            let client_support = ClientSupport::new(client_params, None).unwrap();
            let key_set = client_support.keyset(None, None).unwrap();
            let eval_keys = keySetGetEvaluationKeys(key_set);
            // serialize eval keys
            let serialized_eval_keys = eval_keys.serialize().unwrap();
            let eval_keys = EvaluationKeys::unserialize(&serialized_eval_keys).unwrap();
            // build lambda arguments from scalar and encrypt them
            let args = [lambdaArgumentFromScalar(4), lambdaArgumentFromScalar(2)];
            let encrypted_args = client_support.encrypt_args(&args, key_set).unwrap();
            // free args
            args.map(|arg| lambdaArgumentDestroy(arg));
            // serialize args
            let serialized_encrypted_args = encrypted_args.serialize().unwrap();
            let encrypted_args =
                PublicArguments::unserialize(&serialized_encrypted_args, client_params).unwrap();
            // execute the compiled function on the encrypted arguments
            let encrypted_result = lib_support
                .server_lambda_call(server_lambda, encrypted_args, eval_keys)
                .unwrap();
            // serialize result
            let serialized_encrypted_result = encrypted_result.serialize().unwrap();
            let encrypted_result =
                PublicResult::unserialize(&serialized_encrypted_result, client_params).unwrap();
            // decrypt the result of execution
            let result_arg = client_support
                .decrypt_result(encrypted_result, key_set)
                .unwrap();
            // get the scalar value from the result lambda argument
            let result = lambdaArgumentGetScalar(result_arg);
            assert_eq!(result, 6);
        }
    }

    #[test]
    fn test_tensor_lambda_argument() {
        unsafe {
            let tensor_data = [1, 2, 3, 73u64];
            let tensor_dims = [2, 2i64];
            let tensor_arg =
                lambdaArgumentFromTensorU64(tensor_data.as_ptr(), tensor_dims.as_ptr(), 2);
            assert!(!lambdaArgumentIsNull(tensor_arg));
            assert!(!lambdaArgumentIsScalar(tensor_arg));
            assert!(lambdaArgumentIsTensor(tensor_arg));
            assert_eq!(lambdaArgumentGetTensorRank(tensor_arg), 2);
            assert_eq!(lambdaArgumentGetTensorDataSize(tensor_arg), 4);
            let mut dims: [i64; 2] = [0, 0];
            assert_eq!(
                lambdaArgumentGetTensorDims(tensor_arg, dims.as_mut_ptr()),
                true
            );
            assert_eq!(dims, tensor_dims);

            let mut data: [u64; 4] = [0; 4];
            assert_eq!(
                lambdaArgumentGetTensorData(tensor_arg, data.as_mut_ptr()),
                true
            );
            assert_eq!(data, tensor_data);
            lambdaArgumentDestroy(tensor_arg);
        }
    }

    #[test]
    fn test_compiler_compile_and_exec_tensor_args() {
        unsafe {
            let module_to_compile = "
            func.func @main(%arg0: tensor<2x3x!FHE.eint<5>>, %arg1: tensor<2x3x!FHE.eint<5>>) -> tensor<2x3x!FHE.eint<5>> {
                    %0 = \"FHELinalg.add_eint\"(%arg0, %arg1) : (tensor<2x3x!FHE.eint<5>>, tensor<2x3x!FHE.eint<5>>) -> tensor<2x3x!FHE.eint<5>>
                    return %0 : tensor<2x3x!FHE.eint<5>>
                }";
            let runtime_library_path = match env::var("CONCRETE_COMPILER_BUILD_DIR") {
                Ok(val) => val + "/lib/libConcretelangRuntime.so",
                Err(_e) => "".to_string(),
            };
            let temp_dir = TempDir::new("rust_test_compiler_compile_and_exec_tensor_args").unwrap();
            let lib_support = LibrarySupport::new(
                temp_dir.path().to_str().unwrap(),
                runtime_library_path.as_str(),
            )
            .unwrap();
            // compile
            let result = lib_support.compile(module_to_compile, None).unwrap();
            // loading materials from compilation
            // - server_lambda: used for execution
            // - client_parameters: used for keygen, encryption, and evaluation keys
            let server_lambda = lib_support.load_server_lambda(result).unwrap();
            let client_params = lib_support.load_client_parameters(result).unwrap();
            let client_support = ClientSupport::new(client_params, None).unwrap();
            let key_set = client_support.keyset(None, None).unwrap();
            let eval_keys = keySetGetEvaluationKeys(key_set);
            // build lambda arguments from scalar and encrypt them
            let args = [
                lambdaArgumentFromTensorU8([1, 2, 3, 4, 5, 6].as_ptr(), [2, 3].as_ptr(), 2),
                lambdaArgumentFromTensorU8([1, 4, 7, 4, 2, 9].as_ptr(), [2, 3].as_ptr(), 2),
            ];
            let encrypted_args = client_support.encrypt_args(&args, key_set).unwrap();
            // execute the compiled function on the encrypted arguments
            let encrypted_result = lib_support
                .server_lambda_call(server_lambda, encrypted_args, eval_keys)
                .unwrap();
            // decrypt the result of execution
            let result_arg = client_support
                .decrypt_result(encrypted_result, key_set)
                .unwrap();
            // check the tensor dims value from the result lambda argument
            assert_eq!(lambdaArgumentGetTensorRank(result_arg), 2);
            assert_eq!(lambdaArgumentGetTensorDataSize(result_arg), 6);
            let mut dims = [0, 0];
            assert!(lambdaArgumentGetTensorDims(result_arg, dims.as_mut_ptr()));
            assert_eq!(dims, [2, 3]);
            // check the tensor data from the result lambda argument
            let mut data = [0; 6];
            assert!(lambdaArgumentGetTensorData(result_arg, data.as_mut_ptr()));
            assert_eq!(data, [2, 6, 10, 8, 7, 15]);
        }
    }
}
