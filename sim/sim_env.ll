; ModuleID = 'builtin.module'
source_filename = "sim_env"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

define ptx_kernel void @control({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, { i32, i32, i64, float, float, float, float } %v1, ptr %v2) #0 !dbg !3 {
entry:
  %__cuda_oxide_local_x3109383809627566666572730942756666657273_ = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_ = alloca { i32, i32, i64, float, float, float, float }, align 8
  %v5 = alloca {  }, align 1
  %__cuda_oxide_local_x3138093230097461736b095461736b5374617465_ = alloca { float, float, float, i32, float }, align 4
  %v7 = alloca [4 x float], align 4
  %v8 = alloca [4 x float], align 4
  %__cuda_oxide_local_x343109313609626f6479095b6633323b20345d_ = alloca [4 x float], align 4
  %__cuda_oxide_local_x343209313609626f6479095b6633323b20345d_ = alloca [4 x float], align 4
  br label %bb0
bb0:
  %v11 = phi { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } [ %v0, %entry ]
  %v12 = phi { i32, i32, i64, float, float, float, float } [ %v1, %entry ]
  %v13 = phi ptr [ %v2, %entry ]
  store { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v11, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8
  store { i32, i32, i64, float, float, float, float } %v12, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8
  %v14 = call i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env(ptr %v5) #0, !dbg !5
  br label %bb1, !dbg !5
bb1:
  %v15 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, i32 0, i32 0, !dbg !7
  %v16 = load i32, ptr %v15, align 8, !dbg !7
  %v17 = zext i32 %v16 to i64, !dbg !7
  %v18 = icmp uge i64 %v14, %v17, !dbg !8
  %v19 = xor i1 %v18, 1, !dbg !8
  br i1 %v19, label %bb3, label %bb2, !dbg !8
bb2:
  br label %bb4, !dbg !9
bb3:
  %v20 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 0, !dbg !10
  %v21 = load ptr, ptr %v20, align 8, !dbg !10
  %v22 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v21, i64 %v14, !dbg !18
  %v23 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v22, i32 0, i32 0, !dbg !19
  %v24 = load [4 x float], ptr %v23, align 16, !dbg !19
  store [4 x float] %v24, ptr %__cuda_oxide_local_x343109313609626f6479095b6633323b20345d_, align 4, !dbg !19
  %v25 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v22, i32 0, i32 1, !dbg !19
  %v26 = load [4 x float], ptr %v25, align 16, !dbg !19
  store [4 x float] %v26, ptr %__cuda_oxide_local_x343209313609626f6479095b6633323b20345d_, align 4, !dbg !19
  %v27 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v22, i32 0, i32 2, !dbg !19
  %v28 = load [4 x float], ptr %v27, align 16, !dbg !19
  %v29 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v22, i32 0, i32 3, !dbg !19
  %v30 = load [4 x float], ptr %v29, align 16, !dbg !19
  %v31 = getelementptr inbounds i64, ptr %v13, i64 %v14, !dbg !23
  %v32 = load i64, ptr %v31, align 8, !dbg !24
  %v33 = sitofp i64 %v32 to float, !dbg !25
  %v34 = fsub contract float %v33, 1.0, !dbg !26
  %v35 = fmul contract float %v34, 2.0, !dbg !26
  %v36 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 2, !dbg !27
  %v37 = load ptr, ptr %v36, align 8, !dbg !27
  %v38 = getelementptr inbounds { float, float, float, i32, float }, ptr %v37, i64 %v14, !dbg !18
  %v39 = load { float, float, float, i32, float }, ptr %v38, align 4, !dbg !28
  store { float, float, float, i32, float } %v39, ptr %__cuda_oxide_local_x3138093230097461736b095461736b5374617465_, align 4, !dbg !28
  %v40 = getelementptr inbounds float, ptr %__cuda_oxide_local_x343109313609626f6479095b6633323b20345d_, i64 0, !dbg !29
  %v41 = load float, ptr %v40, align 4, !dbg !29
  %v42 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3138093230097461736b095461736b5374617465_, i32 0, i32 0, !dbg !30
  store float %v41, ptr %v42, align 4, !dbg !30
  %v43 = getelementptr inbounds float, ptr %__cuda_oxide_local_x343109313609626f6479095b6633323b20345d_, i64 2, !dbg !31
  %v44 = load float, ptr %v43, align 4, !dbg !31
  %v45 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3138093230097461736b095461736b5374617465_, i32 0, i32 1, !dbg !32
  store float %v44, ptr %v45, align 4, !dbg !32
  %v46 = fmul contract float %v35, 0.5, !dbg !33
  %v47 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3138093230097461736b095461736b5374617465_, i32 0, i32 4, !dbg !33
  store float %v46, ptr %v47, align 4, !dbg !33
  %v48 = load { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3138093230097461736b095461736b5374617465_, align 4, !dbg !34
  store { float, float, float, i32, float } %v48, ptr %v38, align 4, !dbg !34
  %v49 = getelementptr inbounds float, ptr %__cuda_oxide_local_x343209313609626f6479095b6633323b20345d_, i64 0, !dbg !35
  %v50 = load float, ptr %v49, align 4, !dbg !35
  %v51 = fsub contract float %v35, %v50, !dbg !36
  %v52 = fmul contract float 8.0, %v51, !dbg !37
  %v53 = getelementptr inbounds float, ptr %__cuda_oxide_local_x343109313609626f6479095b6633323b20345d_, i64 1, !dbg !38
  %v54 = load float, ptr %v53, align 4, !dbg !38
  %v55 = fsub contract float 1.2000000476837158, %v54, !dbg !39
  %v56 = fmul contract float 20.0, %v55, !dbg !40
  %v57 = fadd contract float 9.8100004196167, %v56, !dbg !41
  %v58 = getelementptr inbounds float, ptr %__cuda_oxide_local_x343209313609626f6479095b6633323b20345d_, i64 1, !dbg !42
  %v59 = load float, ptr %v58, align 4, !dbg !42
  %v60 = fmul contract float 8.0, %v59, !dbg !43
  %v61 = fsub contract float %v57, %v60, !dbg !41
  %v62 = getelementptr inbounds float, ptr %__cuda_oxide_local_x343209313609626f6479095b6633323b20345d_, i64 2, !dbg !44
  %v63 = load float, ptr %v62, align 4, !dbg !44
  %v64 = fsub contract float -2.0, %v63, !dbg !45
  %v65 = fmul contract float 8.0, %v64, !dbg !46
  %v66 = insertvalue [4 x float] undef, float %v52, 0
  %v67 = insertvalue [4 x float] %v66, float %v61, 1
  %v68 = insertvalue [4 x float] %v67, float %v65, 2
  %v69 = insertvalue [4 x float] %v68, float 0.0, 3, !dbg !47
  %v70 = insertvalue [4 x float] undef, float 0.0, 0
  %v71 = insertvalue [4 x float] %v70, float 0.0, 1
  %v72 = insertvalue [4 x float] %v71, float 0.0, 2
  %v73 = insertvalue [4 x float] %v72, float 0.0, 3, !dbg !48
  %v74 = insertvalue { [4 x float], [4 x float] } undef, [4 x float] %v69, 0
  %v75 = insertvalue { [4 x float], [4 x float] } %v74, [4 x float] %v73, 1, !dbg !49
  %v76 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 1, !dbg !50
  %v77 = load ptr, ptr %v76, align 8, !dbg !50
  %v78 = getelementptr inbounds { [4 x float], [4 x float] }, ptr %v77, i64 %v14, !dbg !18
  store { [4 x float], [4 x float] } %v75, ptr %v78, align 16, !dbg !51
  br label %bb4, !dbg !9
bb4:
  ret void, !dbg !9
}

define ptx_kernel void @reset({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, { i32, i32, i64, float, float, float, float } %v1) #0 !dbg !52 {
entry:
  %__cuda_oxide_local_x3109383809627566666572730942756666657273_ = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_ = alloca { i32, i32, i64, float, float, float, float }, align 8
  %v4 = alloca {  }, align 1
  %v5 = alloca [4 x float], align 4
  br label %bb0
bb0:
  %v6 = phi { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } [ %v0, %entry ]
  %v7 = phi { i32, i32, i64, float, float, float, float } [ %v1, %entry ]
  store { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v6, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8
  store { i32, i32, i64, float, float, float, float } %v7, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8
  %v8 = call i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env(ptr %v4) #0, !dbg !54
  br label %bb1, !dbg !54
bb1:
  %v9 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, i32 0, i32 0, !dbg !56
  %v10 = load i32, ptr %v9, align 8, !dbg !56
  %v11 = zext i32 %v10 to i64, !dbg !56
  %v12 = icmp uge i64 %v8, %v11, !dbg !57
  %v13 = xor i1 %v12, 1, !dbg !57
  br i1 %v13, label %bb3, label %bb2, !dbg !57
bb2:
  br label %bb6, !dbg !58
bb3:
  %v14 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 10, !dbg !59
  %v15 = load ptr, ptr %v14, align 8, !dbg !59
  %v16 = getelementptr inbounds i64, ptr %v15, i64 %v8, !dbg !62
  store i64 0, ptr %v16, align 8, !dbg !63
  %v17 = insertvalue [4 x float] undef, float 0.0, 0
  %v18 = insertvalue [4 x float] %v17, float 0.0, 1
  %v19 = insertvalue [4 x float] %v18, float 0.0, 2
  %v20 = insertvalue [4 x float] %v19, float 0.0, 3, !dbg !64
  %v21 = insertvalue { [4 x float], [4 x float] } undef, [4 x float] %v20, 0
  %v22 = insertvalue { [4 x float], [4 x float] } %v21, [4 x float] %v20, 1, !dbg !65
  %v23 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 1, !dbg !66
  %v24 = load ptr, ptr %v23, align 8, !dbg !66
  %v25 = getelementptr inbounds { [4 x float], [4 x float] }, ptr %v24, i64 %v8, !dbg !62
  store { [4 x float], [4 x float] } %v22, ptr %v25, align 16, !dbg !67
  %v26 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 5, !dbg !68
  %v27 = load ptr, ptr %v26, align 8, !dbg !68
  %v28 = getelementptr inbounds float, ptr %v27, i64 %v8, !dbg !62
  store float 0.0, ptr %v28, align 4, !dbg !69
  %v29 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 6, !dbg !70
  %v30 = load ptr, ptr %v29, align 8, !dbg !70
  %v31 = getelementptr inbounds float, ptr %v30, i64 %v8, !dbg !62
  store float 0.0, ptr %v31, align 4, !dbg !71
  %v32 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 7, !dbg !72
  %v33 = load ptr, ptr %v32, align 8, !dbg !72
  %v34 = getelementptr inbounds float, ptr %v33, i64 %v8, !dbg !62
  store float 0.0, ptr %v34, align 4, !dbg !73
  %v35 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 9, !dbg !74
  %v36 = load ptr, ptr %v35, align 8, !dbg !74
  %v37 = getelementptr inbounds float, ptr %v36, i64 %v8, !dbg !62
  store float 0.0, ptr %v37, align 4, !dbg !75
  %v38 = load { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8, !dbg !76
  %v39 = load { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8, !dbg !76
  %v40 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 0
  %v41 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 1
  %v42 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 2
  %v43 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 3
  %v44 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 4
  %v45 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 5
  %v46 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 6
  %v47 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 7
  %v48 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 8
  %v49 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 9
  %v50 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v38, 10
  %v51 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 0
  %v52 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 1
  %v53 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 2
  %v54 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 3
  %v55 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 4
  %v56 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 5
  %v57 = extractvalue { i32, i32, i64, float, float, float, float } %v39, 6
  call void @sim_env__kernels__device__reset_one(i64 %v8, ptr %v40, ptr %v41, ptr %v42, ptr %v43, ptr %v44, ptr %v45, ptr %v46, ptr %v47, ptr %v48, ptr %v49, ptr %v50, i32 %v51, i32 %v52, i64 %v53, float %v54, float %v55, float %v56, float %v57, i64 0) #0, !dbg !76
  br label %bb4, !dbg !76
bb4:
  %v59 = load { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8, !dbg !77
  %v60 = load { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8, !dbg !77
  %v61 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 0
  %v62 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 1
  %v63 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 2
  %v64 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 3
  %v65 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 4
  %v66 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 5
  %v67 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 6
  %v68 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 7
  %v69 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 8
  %v70 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 9
  %v71 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v59, 10
  %v72 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 0
  %v73 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 1
  %v74 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 2
  %v75 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 3
  %v76 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 4
  %v77 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 5
  %v78 = extractvalue { i32, i32, i64, float, float, float, float } %v60, 6
  call void @sim_env__kernels__device__observe_one(i64 %v8, ptr %v61, ptr %v62, ptr %v63, ptr %v64, ptr %v65, ptr %v66, ptr %v67, ptr %v68, ptr %v69, ptr %v70, ptr %v71, i32 %v72, i32 %v73, i64 %v74, float %v75, float %v76, float %v77, float %v78) #0, !dbg !77
  br label %bb5, !dbg !77
bb5:
  br label %bb6, !dbg !58
bb6:
  ret void, !dbg !58
}

define ptx_kernel void @finish({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, { i32, i32, i64, float, float, float, float } %v1) #0 !dbg !78 {
entry:
  %__cuda_oxide_local_x3109383809627566666572730942756666657273_ = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_ = alloca { i32, i32, i64, float, float, float, float }, align 8
  %v4 = alloca {  }, align 1
  %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_ = alloca { float, float, float, i32, float }, align 4
  %__cuda_oxide_local_x363109313609626f6479095b6633323b20345d_ = alloca [4 x float], align 4
  br label %bb0
bb0:
  %v7 = phi { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } [ %v0, %entry ]
  %v8 = phi { i32, i32, i64, float, float, float, float } [ %v1, %entry ]
  store { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v7, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8
  store { i32, i32, i64, float, float, float, float } %v8, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8
  %v9 = call i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env(ptr %v4) #0, !dbg !80
  br label %bb1, !dbg !80
bb1:
  %v10 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, i32 0, i32 0, !dbg !82
  %v11 = load i32, ptr %v10, align 8, !dbg !82
  %v12 = zext i32 %v11 to i64, !dbg !82
  %v13 = icmp uge i64 %v9, %v12, !dbg !83
  %v14 = xor i1 %v13, 1, !dbg !83
  br i1 %v14, label %bb3, label %bb2, !dbg !83
bb2:
  br label %bb35, !dbg !84
bb3:
  %v15 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 0, !dbg !85
  %v16 = load ptr, ptr %v15, align 8, !dbg !85
  %v17 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v16, i64 %v9, !dbg !95
  %v18 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v17, i32 0, i32 0, !dbg !96
  %v19 = load [4 x float], ptr %v18, align 16, !dbg !96
  store [4 x float] %v19, ptr %__cuda_oxide_local_x363109313609626f6479095b6633323b20345d_, align 4, !dbg !96
  %v20 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v17, i32 0, i32 1, !dbg !96
  %v21 = load [4 x float], ptr %v20, align 16, !dbg !96
  %v22 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v17, i32 0, i32 2, !dbg !96
  %v23 = load [4 x float], ptr %v22, align 16, !dbg !96
  %v24 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v17, i32 0, i32 3, !dbg !96
  %v25 = load [4 x float], ptr %v24, align 16, !dbg !96
  %v26 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 2, !dbg !97
  %v27 = load ptr, ptr %v26, align 8, !dbg !97
  %v28 = getelementptr inbounds { float, float, float, i32, float }, ptr %v27, i64 %v9, !dbg !95
  %v29 = load { float, float, float, i32, float }, ptr %v28, align 4, !dbg !98
  store { float, float, float, i32, float } %v29, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, align 4, !dbg !98
  %v30 = getelementptr inbounds float, ptr %__cuda_oxide_local_x363109313609626f6479095b6633323b20345d_, i64 0, !dbg !99
  %v31 = load float, ptr %v30, align 4, !dbg !99
  %v32 = uitofp i64 %v9 to float, !dbg !100
  %v33 = fmul contract float %v32, 32.0, !dbg !100
  %v34 = fsub contract float %v31, %v33, !dbg !99
  %v35 = fcmp olt float %v34, -3.819999933242798, !dbg !101
  %v36 = xor i1 %v35, 1, !dbg !101
  br i1 %v36, label %bb4, label %bb5, !dbg !101
bb4:
  %v37 = fcmp ogt float %v34, 3.819999933242798, !dbg !102
  %v38 = xor i1 %v37, 1, !dbg !102
  br i1 %v38, label %bb6, label %bb5, !dbg !102
bb5:
  br label %bb8, !dbg !101
bb6:
  %v39 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 0, !dbg !103
  %v40 = load float, ptr %v39, align 4, !dbg !103
  %v41 = fsub contract float %v40, %v33, !dbg !103
  %v42 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 1, !dbg !104
  %v43 = load float, ptr %v42, align 4, !dbg !104
  %v44 = getelementptr inbounds float, ptr %__cuda_oxide_local_x363109313609626f6479095b6633323b20345d_, i64 2, !dbg !105
  %v45 = load float, ptr %v44, align 4, !dbg !105
  %v46 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 2, !dbg !106
  %v47 = load float, ptr %v46, align 4, !dbg !106
  %v48 = call i1 @sim_env__task__hits_wall(float %v41, float %v43, float %v34, float %v45, float %v47) #0, !dbg !107
  br label %bb7, !dbg !107
bb7:
  br label %bb8, !dbg !101
bb8:
  %v49 = phi i1 [ 1, %bb5 ], [ %v48, %bb7 ]
  %v50 = xor i1 %v49, 1, !dbg !108
  br i1 %v50, label %bb9, label %bb10, !dbg !108
bb9:
  %v51 = getelementptr inbounds float, ptr %__cuda_oxide_local_x363109313609626f6479095b6633323b20345d_, i64 2, !dbg !109
  %v52 = load float, ptr %v51, align 4, !dbg !109
  %v53 = fcmp ole float %v52, -12.0, !dbg !109
  br label %bb11, !dbg !110
bb10:
  br label %bb11, !dbg !110
bb11:
  %v54 = phi i1 [ %v53, %bb9 ], [ 0, %bb10 ]
  %v55 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 3, !dbg !111
  %v56 = load i32, ptr %v55, align 4, !dbg !111
  %v57 = add i32 %v56, 1, !dbg !111
  %v58 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 3, !dbg !111
  store i32 %v57, ptr %v58, align 4, !dbg !111
  %v59 = xor i1 %v49, 1, !dbg !112
  br i1 %v59, label %bb12, label %bb13, !dbg !112
bb12:
  %v60 = xor i1 %v54, 1, !dbg !113
  br i1 %v60, label %bb14, label %bb13, !dbg !113
bb13:
  br label %bb15, !dbg !112
bb14:
  %v61 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 3, !dbg !114
  %v62 = load i32, ptr %v61, align 4, !dbg !114
  %v63 = icmp uge i32 %v62, 150, !dbg !114
  br label %bb15, !dbg !112
bb15:
  %v64 = phi i1 [ 1, %bb13 ], [ %v63, %bb14 ]
  %v65 = getelementptr inbounds { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, i32 0, i32 1, !dbg !115
  %v66 = load float, ptr %v65, align 4, !dbg !115
  %v67 = getelementptr inbounds float, ptr %__cuda_oxide_local_x363109313609626f6479095b6633323b20345d_, i64 2, !dbg !116
  %v68 = load float, ptr %v67, align 4, !dbg !116
  %v69 = fsub contract float %v66, %v68, !dbg !117
  %v70 = fmul contract float %v69, 0.20000000298023224, !dbg !117
  %v71 = fsub contract float %v70, 0.009999999776482582, !dbg !117
  %v72 = xor i1 %v54, 1, !dbg !118
  br i1 %v72, label %bb17, label %bb16, !dbg !118
bb16:
  br label %bb21, !dbg !119
bb17:
  %v73 = xor i1 %v49, 1, !dbg !120
  br i1 %v73, label %bb19, label %bb18, !dbg !120
bb18:
  br label %bb20, !dbg !121
bb19:
  br label %bb20, !dbg !121
bb20:
  %v74 = phi float [ -5.0, %bb18 ], [ 0.0, %bb19 ]
  br label %bb21, !dbg !119
bb21:
  %v75 = phi float [ 5.0, %bb16 ], [ %v74, %bb20 ]
  %v76 = fadd contract float %v71, %v75, !dbg !117
  %v77 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 5, !dbg !122
  %v78 = load ptr, ptr %v77, align 8, !dbg !122
  %v79 = getelementptr inbounds float, ptr %v78, i64 %v9, !dbg !95
  store float %v76, ptr %v79, align 4, !dbg !123
  %v80 = xor i1 %v64, 1, !dbg !124
  br i1 %v80, label %bb23, label %bb22, !dbg !124
bb22:
  br label %bb24, !dbg !125
bb23:
  br label %bb24, !dbg !125
bb24:
  %v81 = phi float [ 1.0, %bb22 ], [ 0.0, %bb23 ]
  %v82 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 6, !dbg !126
  %v83 = load ptr, ptr %v82, align 8, !dbg !126
  %v84 = getelementptr inbounds float, ptr %v83, i64 %v9, !dbg !95
  store float %v81, ptr %v84, align 4, !dbg !127
  %v85 = xor i1 %v54, 1, !dbg !128
  br i1 %v85, label %bb26, label %bb25, !dbg !128
bb25:
  br label %bb27, !dbg !129
bb26:
  br label %bb27, !dbg !129
bb27:
  %v86 = phi float [ 1.0, %bb25 ], [ 0.0, %bb26 ]
  %v87 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 7, !dbg !130
  %v88 = load ptr, ptr %v87, align 8, !dbg !130
  %v89 = getelementptr inbounds float, ptr %v88, i64 %v9, !dbg !95
  store float %v86, ptr %v89, align 4, !dbg !131
  %v90 = xor i1 %v49, 1, !dbg !132
  br i1 %v90, label %bb29, label %bb28, !dbg !132
bb28:
  br label %bb30, !dbg !133
bb29:
  br label %bb30, !dbg !133
bb30:
  %v91 = phi float [ 1.0, %bb28 ], [ 0.0, %bb29 ]
  %v92 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 9, !dbg !134
  %v93 = load ptr, ptr %v92, align 8, !dbg !134
  %v94 = getelementptr inbounds float, ptr %v93, i64 %v9, !dbg !95
  store float %v91, ptr %v94, align 4, !dbg !135
  %v95 = load { float, float, float, i32, float }, ptr %__cuda_oxide_local_x3132093230097461736b095461736b5374617465_, align 4, !dbg !136
  store { float, float, float, i32, float } %v95, ptr %v28, align 4, !dbg !136
  %v96 = xor i1 %v64, 1, !dbg !137
  br i1 %v96, label %bb33, label %bb31, !dbg !137
bb31:
  %v97 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, i32 0, i32 10, !dbg !138
  %v98 = load ptr, ptr %v97, align 8, !dbg !138
  %v99 = getelementptr inbounds i64, ptr %v98, i64 %v9, !dbg !95
  %v100 = load i64, ptr %v99, align 8, !dbg !139
  %v101 = add i64 %v100, 1, !dbg !145
  store i64 %v101, ptr %v99, align 8, !dbg !147
  %v102 = load { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8, !dbg !148
  %v103 = load { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8, !dbg !148
  %v104 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 0
  %v105 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 1
  %v106 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 2
  %v107 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 3
  %v108 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 4
  %v109 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 5
  %v110 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 6
  %v111 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 7
  %v112 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 8
  %v113 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 9
  %v114 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v102, 10
  %v115 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 0
  %v116 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 1
  %v117 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 2
  %v118 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 3
  %v119 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 4
  %v120 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 5
  %v121 = extractvalue { i32, i32, i64, float, float, float, float } %v103, 6
  call void @sim_env__kernels__device__reset_one(i64 %v9, ptr %v104, ptr %v105, ptr %v106, ptr %v107, ptr %v108, ptr %v109, ptr %v110, ptr %v111, ptr %v112, ptr %v113, ptr %v114, i32 %v115, i32 %v116, i64 %v117, float %v118, float %v119, float %v120, float %v121, i64 %v101) #0, !dbg !148
  br label %bb32, !dbg !148
bb32:
  br label %bb33, !dbg !149
bb33:
  %v123 = load { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3109383809627566666572730942756666657273_, align 8, !dbg !150
  %v124 = load { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3209333209636f6e66696709436f6e666967_, align 8, !dbg !150
  %v125 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 0
  %v126 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 1
  %v127 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 2
  %v128 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 3
  %v129 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 4
  %v130 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 5
  %v131 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 6
  %v132 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 7
  %v133 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 8
  %v134 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 9
  %v135 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v123, 10
  %v136 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 0
  %v137 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 1
  %v138 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 2
  %v139 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 3
  %v140 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 4
  %v141 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 5
  %v142 = extractvalue { i32, i32, i64, float, float, float, float } %v124, 6
  call void @sim_env__kernels__device__observe_one(i64 %v9, ptr %v125, ptr %v126, ptr %v127, ptr %v128, ptr %v129, ptr %v130, ptr %v131, ptr %v132, ptr %v133, ptr %v134, ptr %v135, i32 %v136, i32 %v137, i64 %v138, float %v139, float %v140, float %v141, float %v142) #0, !dbg !150
  br label %bb34, !dbg !150
bb34:
  br label %bb35, !dbg !84
bb35:
  ret void, !dbg !84
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()

define i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env(ptr %v0) alwaysinline #0 !dbg !152 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0, !dbg !153
  br label %bb1, !dbg !153
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0, !dbg !154
  br label %bb2, !dbg !154
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0, !dbg !155
  br label %bb7, !dbg !155
bb3:
  %v5 = xor i1 %v20, 1, !dbg !157
  br i1 %v5, label %bb5, label %bb4, !dbg !157
bb4:
  %v6 = icmp ne i64 %v19, 18446744073709551615, !dbg !158
  br label %bb6, !dbg !157
bb5:
  br label %bb6, !dbg !157
bb6:
  %v7 = phi i1 [ %v6, %bb4 ], [ 0, %bb5 ]
  %v8 = xor i1 %v7, 1, !dbg !162
  br i1 %v8, label %bb14, label %bb13, !dbg !162
bb7:
  %v9 = zext i32 %v2 to i64, !dbg !165
  %v10 = zext i32 %v3 to i64, !dbg !166
  %v11 = zext i32 %v4 to i64, !dbg !167
  %v12 = icmp eq i64 %v10, 0, !dbg !168
  br i1 %v12, label %bb10, label %bb8, !dbg !168
bb8:
  %v13 = sub i64 18446744073709551615, %v11, !dbg !169
  %v14 = udiv i64 %v13, %v10, !dbg !169
  %v15 = icmp ugt i64 %v9, %v14, !dbg !170
  %v16 = xor i1 %v15, 1, !dbg !170
  br i1 %v16, label %bb11, label %bb9, !dbg !170
bb9:
  br label %bb10, !dbg !171
bb10:
  br label %bb12, !dbg !172
bb11:
  %v17 = mul i64 %v9, %v10, !dbg !173
  %v18 = add i64 %v17, %v11, !dbg !173
  br label %bb12, !dbg !172
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env(ptr %v1) #0, !dbg !157
  br label %bb3, !dbg !157
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615, !dbg !174
  br i1 %v21, label %bb14, label %bb15, !dbg !174
bb14:
  br label %bb15, !dbg !175
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22, !dbg !176
}

define void @sim_env__kernels__device__reset_one(i64 %v0, ptr %v1, ptr %v2, ptr %v3, ptr %v4, ptr %v5, ptr %v6, ptr %v7, ptr %v8, ptr %v9, ptr %v10, ptr %v11, i32 %v12, i32 %v13, i64 %v14, float %v15, float %v16, float %v17, float %v18, i64 %v19) #0 !dbg !177 {
entry:
  %v20 = alloca [4 x float], align 4
  %v21 = alloca [4 x float], align 4
  %v22 = alloca [4 x float], align 4
  %v23 = alloca [4 x float], align 4
  %__cuda_oxide_local_x32310936340977616c6c09426f64795374617465_ = alloca { [4 x float], [4 x float], [4 x float], [4 x float] }, align 16
  %v25 = alloca [4 x float], align 4
  %v26 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } undef, ptr %v1, 0
  %v27 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v26, ptr %v2, 1
  %v28 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v27, ptr %v3, 2
  %v29 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v28, ptr %v4, 3
  %v30 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v29, ptr %v5, 4
  %v31 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v30, ptr %v6, 5
  %v32 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v31, ptr %v7, 6
  %v33 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v32, ptr %v8, 7
  %v34 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v33, ptr %v9, 8
  %v35 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v34, ptr %v10, 9
  %v36 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v35, ptr %v11, 10
  %v37 = insertvalue { i32, i32, i64, float, float, float, float } undef, i32 %v12, 0
  %v38 = insertvalue { i32, i32, i64, float, float, float, float } %v37, i32 %v13, 1
  %v39 = insertvalue { i32, i32, i64, float, float, float, float } %v38, i64 %v14, 2
  %v40 = insertvalue { i32, i32, i64, float, float, float, float } %v39, float %v15, 3
  %v41 = insertvalue { i32, i32, i64, float, float, float, float } %v40, float %v16, 4
  %v42 = insertvalue { i32, i32, i64, float, float, float, float } %v41, float %v17, 5
  %v43 = insertvalue { i32, i32, i64, float, float, float, float } %v42, float %v18, 6
  br label %bb0
bb0:
  %v44 = phi i64 [ %v0, %entry ]
  %v45 = phi { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } [ %v36, %entry ]
  %v46 = phi { i32, i32, i64, float, float, float, float } [ %v43, %entry ]
  %v47 = phi i64 [ %v19, %entry ]
  %v48 = uitofp i64 %v44 to float, !dbg !178
  %v49 = fmul contract float %v48, 32.0, !dbg !178
  %v50 = extractvalue { i32, i32, i64, float, float, float, float } %v46, 2, !dbg !180
  %v51 = trunc i64 %v44 to i32, !dbg !181
  %v52 = extractvalue { i32, i32, i64, float, float, float, float } %v46, 1, !dbg !182
  %v53 = zext i32 %v51 to i64, !dbg !186
  %v54 = mul i64 %v53, 11400714819323198485, !dbg !190
  %v55 = xor i64 %v50, %v54, !dbg !191
  %v56 = mul i64 %v47, 15111065706836454659, !dbg !190
  %v57 = xor i64 %v55, %v56, !dbg !191
  %v58 = zext i32 %v52 to i64, !dbg !192
  %v59 = mul i64 %v58, 10723151780598845931, !dbg !190
  %v60 = xor i64 %v57, %v59, !dbg !191
  %v61 = zext i32 30 to i64
  %v62 = and i64 %v61, 63
  %v63 = lshr i64 %v60, %v62, !dbg !193
  %v64 = xor i64 %v60, %v63, !dbg !194
  %v65 = mul i64 %v64, 13787848793156543929, !dbg !190
  %v66 = zext i32 27 to i64
  %v67 = and i64 %v66, 63
  %v68 = lshr i64 %v65, %v67, !dbg !195
  %v69 = xor i64 %v65, %v68, !dbg !196
  %v70 = mul i64 %v69, 10723151780598845931, !dbg !190
  %v71 = zext i32 31 to i64
  %v72 = and i64 %v71, 63
  %v73 = lshr i64 %v70, %v72, !dbg !197
  %v74 = xor i64 %v70, %v73, !dbg !198
  %v75 = zext i32 40 to i64
  %v76 = and i64 %v75, 63
  %v77 = lshr i64 %v74, %v76, !dbg !199
  %v78 = uitofp i64 %v77 to float, !dbg !199
  %v79 = fmul contract float %v78, 0.0000002980232238769531, !dbg !199
  %v80 = fsub contract float %v79, 2.5, !dbg !199
  %v81 = insertvalue [4 x float] undef, float %v49, 0
  %v82 = insertvalue [4 x float] %v81, float 1.2000000476837158, 1
  %v83 = insertvalue [4 x float] %v82, float 0.0, 2
  %v84 = insertvalue [4 x float] %v83, float 0.0, 3, !dbg !201
  %v85 = insertvalue [4 x float] undef, float 0.0, 0
  %v86 = insertvalue [4 x float] %v85, float 0.0, 1
  %v87 = insertvalue [4 x float] %v86, float -2.0, 2
  %v88 = insertvalue [4 x float] %v87, float 0.0, 3, !dbg !202
  %v89 = insertvalue [4 x float] undef, float 0.0, 0
  %v90 = insertvalue [4 x float] %v89, float 0.0, 1
  %v91 = insertvalue [4 x float] %v90, float 0.0, 2
  %v92 = insertvalue [4 x float] %v91, float 1.0, 3, !dbg !203
  %v93 = insertvalue [4 x float] undef, float 0.0, 0
  %v94 = insertvalue [4 x float] %v93, float 0.0, 1
  %v95 = insertvalue [4 x float] %v94, float 0.0, 2
  %v96 = insertvalue [4 x float] %v95, float 0.0, 3, !dbg !204
  %v97 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } undef, [4 x float] %v84, 0
  %v98 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } %v97, [4 x float] %v88, 1
  %v99 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } %v98, [4 x float] %v92, 2
  %v100 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } %v99, [4 x float] %v96, 3, !dbg !205
  %v101 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v45, 0, !dbg !206
  %v102 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v101, i64 %v44, !dbg !211
  store { [4 x float], [4 x float], [4 x float], [4 x float] } %v100, ptr %v102, align 16, !dbg !212
  %v103 = insertvalue { float, float, float, i32, float } undef, float %v49, 0
  %v104 = insertvalue { float, float, float, i32, float } %v103, float 0.0, 1
  %v105 = insertvalue { float, float, float, i32, float } %v104, float %v80, 2
  %v106 = insertvalue { float, float, float, i32, float } %v105, i32 0, 3
  %v107 = insertvalue { float, float, float, i32, float } %v106, float 0.0, 4, !dbg !213
  %v108 = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v45, 2, !dbg !214
  %v109 = getelementptr inbounds { float, float, float, i32, float }, ptr %v108, i64 %v44, !dbg !211
  store { float, float, float, i32, float } %v107, ptr %v109, align 4, !dbg !215
  %v110 = fadd contract float %v49, %v80, !dbg !216
  %v111 = fsub contract float %v110, 0.800000011920929, !dbg !216
  %v112 = fsub contract float %v111, 4.0, !dbg !216
  %v113 = insertvalue [4 x float] undef, float %v112, 0
  %v114 = insertvalue [4 x float] %v113, float 1.5, 1
  %v115 = insertvalue [4 x float] %v114, float -6.0, 2
  %v116 = insertvalue [4 x float] %v115, float 0.0, 3, !dbg !217
  %v117 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } undef, [4 x float] %v116, 0
  %v118 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } %v117, [4 x float] %v96, 1
  %v119 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } %v118, [4 x float] %v92, 2
  %v120 = insertvalue { [4 x float], [4 x float], [4 x float], [4 x float] } %v119, [4 x float] %v96, 3, !dbg !218
  store { [4 x float], [4 x float], [4 x float], [4 x float] } %v120, ptr %__cuda_oxide_local_x32310936340977616c6c09426f64795374617465_, align 16, !dbg !218
  %v121 = extractvalue { i32, i32, i64, float, float, float, float } %v46, 0, !dbg !219
  %v122 = zext i32 %v121 to i64, !dbg !219
  %v123 = mul i64 2, %v44, !dbg !220
  %v124 = add i64 %v122, %v123, !dbg !219
  %v125 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v101, i64 %v124, !dbg !211
  %v126 = load { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x32310936340977616c6c09426f64795374617465_, align 16, !dbg !221
  store { [4 x float], [4 x float], [4 x float], [4 x float] } %v126, ptr %v125, align 16, !dbg !221
  %v127 = fadd contract float %v110, 0.800000011920929, !dbg !222
  %v128 = fadd contract float %v127, 4.0, !dbg !223
  %v129 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x32310936340977616c6c09426f64795374617465_, i32 0, i32 0, !dbg !223
  %v130 = getelementptr inbounds float, ptr %v129, i64 0, !dbg !223
  store float %v128, ptr %v130, align 16, !dbg !223
  %v131 = add i64 %v124, 1, !dbg !224
  %v132 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v101, i64 %v131, !dbg !211
  %v133 = load { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x32310936340977616c6c09426f64795374617465_, align 16, !dbg !225
  store { [4 x float], [4 x float], [4 x float], [4 x float] } %v133, ptr %v132, align 16, !dbg !225
  ret void, !dbg !226
}

declare void @llvm.trap()

define void @sim_env__kernels__device__observe_one(i64 %v0, ptr %v1, ptr %v2, ptr %v3, ptr %v4, ptr %v5, ptr %v6, ptr %v7, ptr %v8, ptr %v9, ptr %v10, ptr %v11, i32 %v12, i32 %v13, i64 %v14, float %v15, float %v16, float %v17, float %v18) #0 !dbg !227 {
entry:
  %__cuda_oxide_local_x3209383809627566666572730942756666657273_ = alloca { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, align 8
  %__cuda_oxide_local_x3309333209636f6e66696709436f6e666967_ = alloca { i32, i32, i64, float, float, float, float }, align 8
  %__cuda_oxide_local_x31350933320970726f7072696f095b6633323b20385d_ = alloca [8 x float], align 4
  %__cuda_oxide_local_x32330931360970726976696c65676564095b6633323b20345d_ = alloca [4 x float], align 4
  %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_ = alloca [36 x float], align 4
  %__cuda_oxide_local_x373209313609626f6479095b6633323b20345d_ = alloca [4 x float], align 4
  %__cuda_oxide_local_x373309313609626f6479095b6633323b20345d_ = alloca [4 x float], align 4
  %v26 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } undef, ptr %v1, 0
  %v27 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v26, ptr %v2, 1
  %v28 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v27, ptr %v3, 2
  %v29 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v28, ptr %v4, 3
  %v30 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v29, ptr %v5, 4
  %v31 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v30, ptr %v6, 5
  %v32 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v31, ptr %v7, 6
  %v33 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v32, ptr %v8, 7
  %v34 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v33, ptr %v9, 8
  %v35 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v34, ptr %v10, 9
  %v36 = insertvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v35, ptr %v11, 10
  %v37 = insertvalue { i32, i32, i64, float, float, float, float } undef, i32 %v12, 0
  %v38 = insertvalue { i32, i32, i64, float, float, float, float } %v37, i32 %v13, 1
  %v39 = insertvalue { i32, i32, i64, float, float, float, float } %v38, i64 %v14, 2
  %v40 = insertvalue { i32, i32, i64, float, float, float, float } %v39, float %v15, 3
  %v41 = insertvalue { i32, i32, i64, float, float, float, float } %v40, float %v16, 4
  %v42 = insertvalue { i32, i32, i64, float, float, float, float } %v41, float %v17, 5
  %v43 = insertvalue { i32, i32, i64, float, float, float, float } %v42, float %v18, 6
  br label %bb0
bb0:
  %v44 = phi i64 [ %v0, %entry ]
  %v45 = phi { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } [ %v36, %entry ]
  %v46 = phi { i32, i32, i64, float, float, float, float } [ %v43, %entry ]
  store { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v45, ptr %__cuda_oxide_local_x3209383809627566666572730942756666657273_, align 8
  store { i32, i32, i64, float, float, float, float } %v46, ptr %__cuda_oxide_local_x3309333209636f6e66696709436f6e666967_, align 8
  %v47 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3209383809627566666572730942756666657273_, i32 0, i32 0, !dbg !228
  %v48 = load ptr, ptr %v47, align 8, !dbg !228
  %v49 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v48, i64 %v44, !dbg !243
  %v50 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v49, i32 0, i32 0, !dbg !244
  %v51 = load [4 x float], ptr %v50, align 16, !dbg !244
  store [4 x float] %v51, ptr %__cuda_oxide_local_x373209313609626f6479095b6633323b20345d_, align 4, !dbg !244
  %v52 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v49, i32 0, i32 1, !dbg !244
  %v53 = load [4 x float], ptr %v52, align 16, !dbg !244
  store [4 x float] %v53, ptr %__cuda_oxide_local_x373309313609626f6479095b6633323b20345d_, align 4, !dbg !244
  %v54 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v49, i32 0, i32 2, !dbg !244
  %v55 = load [4 x float], ptr %v54, align 16, !dbg !244
  %v56 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v49, i32 0, i32 3, !dbg !244
  %v57 = load [4 x float], ptr %v56, align 16, !dbg !244
  %v58 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3209383809627566666572730942756666657273_, i32 0, i32 2, !dbg !245
  %v59 = load ptr, ptr %v58, align 8, !dbg !245
  %v60 = getelementptr inbounds { float, float, float, i32, float }, ptr %v59, i64 %v44, !dbg !243
  %v61 = getelementptr inbounds { float, float, float, i32, float }, ptr %v60, i32 0, i32 0, !dbg !246
  %v62 = load float, ptr %v61, align 4, !dbg !246
  %v63 = getelementptr inbounds { float, float, float, i32, float }, ptr %v60, i32 0, i32 1, !dbg !246
  %v64 = load float, ptr %v63, align 4, !dbg !246
  %v65 = getelementptr inbounds { float, float, float, i32, float }, ptr %v60, i32 0, i32 2, !dbg !246
  %v66 = load float, ptr %v65, align 4, !dbg !246
  %v67 = getelementptr inbounds { float, float, float, i32, float }, ptr %v60, i32 0, i32 3, !dbg !246
  %v68 = load i32, ptr %v67, align 4, !dbg !246
  %v69 = getelementptr inbounds { float, float, float, i32, float }, ptr %v60, i32 0, i32 4, !dbg !246
  %v70 = load float, ptr %v69, align 4, !dbg !246
  %v71 = getelementptr inbounds float, ptr %__cuda_oxide_local_x373209313609626f6479095b6633323b20345d_, i64 0, !dbg !247
  %v72 = load float, ptr %v71, align 4, !dbg !247
  %v73 = getelementptr inbounds float, ptr %__cuda_oxide_local_x373209313609626f6479095b6633323b20345d_, i64 1, !dbg !248
  %v74 = load float, ptr %v73, align 4, !dbg !248
  %v75 = getelementptr inbounds float, ptr %__cuda_oxide_local_x373209313609626f6479095b6633323b20345d_, i64 2, !dbg !249
  %v76 = load float, ptr %v75, align 4, !dbg !249
  %v77 = uitofp i64 %v44 to float, !dbg !250
  %v78 = fmul contract float %v77, 32.0, !dbg !250
  %v79 = uitofp i32 %v68 to float, !dbg !251
  %v80 = fdiv contract float %v79, 150.0, !dbg !251
  %v81 = fsub contract float %v78, %v72, !dbg !252
  %v82 = fsub contract float 1.2000000476837158, %v74, !dbg !253
  %v83 = fsub contract float -12.0, %v76, !dbg !254
  %v84 = getelementptr inbounds float, ptr %__cuda_oxide_local_x373309313609626f6479095b6633323b20345d_, i64 0, !dbg !255
  %v85 = load float, ptr %v84, align 4, !dbg !255
  %v86 = getelementptr inbounds float, ptr %__cuda_oxide_local_x373309313609626f6479095b6633323b20345d_, i64 1, !dbg !256
  %v87 = load float, ptr %v86, align 4, !dbg !256
  %v88 = getelementptr inbounds float, ptr %__cuda_oxide_local_x373309313609626f6479095b6633323b20345d_, i64 2, !dbg !257
  %v89 = load float, ptr %v88, align 4, !dbg !257
  %v90 = fsub contract float 1.0, %v80, !dbg !258
  %v91 = insertvalue [8 x float] undef, float %v81, 0
  %v92 = insertvalue [8 x float] %v91, float %v82, 1
  %v93 = insertvalue [8 x float] %v92, float %v83, 2
  %v94 = insertvalue [8 x float] %v93, float %v85, 3
  %v95 = insertvalue [8 x float] %v94, float %v87, 4
  %v96 = insertvalue [8 x float] %v95, float %v89, 5
  %v97 = insertvalue [8 x float] %v96, float %v90, 6
  %v98 = insertvalue [8 x float] %v97, float %v70, 7, !dbg !259
  store [8 x float] %v98, ptr %__cuda_oxide_local_x31350933320970726f7072696f095b6633323b20385d_, align 4, !dbg !259
  %v99 = fadd contract float %v78, %v66, !dbg !260
  %v100 = fsub contract float %v99, %v72, !dbg !260
  %v101 = fsub contract float -6.0, %v76, !dbg !261
  %v102 = insertvalue [4 x float] undef, float %v100, 0
  %v103 = insertvalue [4 x float] %v102, float %v101, 1
  %v104 = insertvalue [4 x float] %v103, float 0.800000011920929, 2
  %v105 = insertvalue [4 x float] %v104, float %v80, 3, !dbg !262
  store [4 x float] %v105, ptr %__cuda_oxide_local_x32330931360970726976696c65676564095b6633323b20345d_, align 4, !dbg !262
  br label %bb1, !dbg !263
bb1:
  %v106 = phi i64 [ 0, %bb0 ], [ %v117, %bb4 ]
  %v107 = icmp ult i64 %v106, 8, !dbg !264
  %v108 = xor i1 %v107, 1, !dbg !264
  br i1 %v108, label %bb5, label %bb2, !dbg !264
bb2:
  %v109 = icmp ult i64 %v106, 8, !dbg !265
  br i1 %v109, label %bb3, label %bb16, !dbg !265
bb3:
  br label %bb4, !dbg !265
bb4:
  %v110 = getelementptr inbounds float, ptr %__cuda_oxide_local_x31350933320970726f7072696f095b6633323b20385d_, i64 %v106, !dbg !265
  %v111 = load float, ptr %v110, align 4, !dbg !265
  %v112 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3209383809627566666572730942756666657273_, i32 0, i32 4, !dbg !266
  %v113 = load ptr, ptr %v112, align 8, !dbg !266
  %v114 = mul i64 %v44, 8, !dbg !267
  %v115 = add i64 %v114, %v106, !dbg !267
  %v116 = getelementptr inbounds float, ptr %v113, i64 %v115, !dbg !243
  store float %v111, ptr %v116, align 4, !dbg !268
  %v117 = add i64 %v106, 1, !dbg !269
  br label %bb1, !dbg !263
bb5:
  br label %bb6, !dbg !270
bb6:
  %v118 = phi i64 [ 0, %bb5 ], [ %v129, %bb9 ]
  %v119 = icmp ult i64 %v118, 4, !dbg !271
  %v120 = xor i1 %v119, 1, !dbg !271
  br i1 %v120, label %bb10, label %bb7, !dbg !271
bb7:
  %v121 = icmp ult i64 %v118, 4, !dbg !272
  br i1 %v121, label %bb8, label %bb17, !dbg !272
bb8:
  br label %bb9, !dbg !272
bb9:
  %v122 = getelementptr inbounds float, ptr %__cuda_oxide_local_x32330931360970726976696c65676564095b6633323b20345d_, i64 %v118, !dbg !272
  %v123 = load float, ptr %v122, align 4, !dbg !272
  %v124 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3209383809627566666572730942756666657273_, i32 0, i32 8, !dbg !273
  %v125 = load ptr, ptr %v124, align 8, !dbg !273
  %v126 = mul i64 %v44, 4, !dbg !274
  %v127 = add i64 %v126, %v118, !dbg !274
  %v128 = getelementptr inbounds float, ptr %v125, i64 %v127, !dbg !243
  store float %v123, ptr %v128, align 4, !dbg !275
  %v129 = add i64 %v118, 1, !dbg !276
  br label %bb6, !dbg !270
bb10:
  %v130 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3309333209636f6e66696709436f6e666967_, i32 0, i32 3, !dbg !277
  %v131 = load float, ptr %v130, align 8, !dbg !277
  %v132 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3309333209636f6e66696709436f6e666967_, i32 0, i32 4, !dbg !278
  %v133 = load float, ptr %v132, align 4, !dbg !278
  %v134 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3309333209636f6e66696709436f6e666967_, i32 0, i32 5, !dbg !279
  %v135 = load float, ptr %v134, align 8, !dbg !279
  %v136 = getelementptr inbounds { i32, i32, i64, float, float, float, float }, ptr %__cuda_oxide_local_x3309333209636f6e66696709436f6e666967_, i32 0, i32 6, !dbg !280
  %v137 = load float, ptr %v136, align 4, !dbg !280
  %v138 = fneg float %v72, !dbg !281
  %v139 = fneg float %v74, !dbg !282
  %v140 = fneg float %v76, !dbg !283
  %v141 = fneg float %v131, !dbg !284
  %v142 = fmul contract float %v141, %v72, !dbg !284
  %v143 = fneg float %v133, !dbg !285
  %v144 = fmul contract float %v143, %v74, !dbg !285
  %v145 = fmul contract float %v135, %v76, !dbg !286
  %v146 = fsub contract float %v137, %v145, !dbg !287
  %v147 = insertvalue [36 x float] undef, float 1.0, 0
  %v148 = insertvalue [36 x float] %v147, float 0.0, 1
  %v149 = insertvalue [36 x float] %v148, float 0.0, 2
  %v150 = insertvalue [36 x float] %v149, float 0.0, 3
  %v151 = insertvalue [36 x float] %v150, float 0.0, 4
  %v152 = insertvalue [36 x float] %v151, float 1.0, 5
  %v153 = insertvalue [36 x float] %v152, float 0.0, 6
  %v154 = insertvalue [36 x float] %v153, float 0.0, 7
  %v155 = insertvalue [36 x float] %v154, float 0.0, 8
  %v156 = insertvalue [36 x float] %v155, float 0.0, 9
  %v157 = insertvalue [36 x float] %v156, float 1.0, 10
  %v158 = insertvalue [36 x float] %v157, float 0.0, 11
  %v159 = insertvalue [36 x float] %v158, float %v138, 12
  %v160 = insertvalue [36 x float] %v159, float %v139, 13
  %v161 = insertvalue [36 x float] %v160, float %v140, 14
  %v162 = insertvalue [36 x float] %v161, float 1.0, 15
  %v163 = insertvalue [36 x float] %v162, float %v131, 16
  %v164 = insertvalue [36 x float] %v163, float 0.0, 17
  %v165 = insertvalue [36 x float] %v164, float 0.0, 18
  %v166 = insertvalue [36 x float] %v165, float 0.0, 19
  %v167 = insertvalue [36 x float] %v166, float 0.0, 20
  %v168 = insertvalue [36 x float] %v167, float %v133, 21
  %v169 = insertvalue [36 x float] %v168, float 0.0, 22
  %v170 = insertvalue [36 x float] %v169, float 0.0, 23
  %v171 = insertvalue [36 x float] %v170, float 0.0, 24
  %v172 = insertvalue [36 x float] %v171, float 0.0, 25
  %v173 = insertvalue [36 x float] %v172, float %v135, 26
  %v174 = insertvalue [36 x float] %v173, float -1.0, 27
  %v175 = insertvalue [36 x float] %v174, float %v142, 28
  %v176 = insertvalue [36 x float] %v175, float %v144, 29
  %v177 = insertvalue [36 x float] %v176, float %v146, 30
  %v178 = insertvalue [36 x float] %v177, float %v76, 31
  %v179 = insertvalue [36 x float] %v178, float %v72, 32
  %v180 = insertvalue [36 x float] %v179, float %v74, 33
  %v181 = insertvalue [36 x float] %v180, float %v76, 34
  %v182 = insertvalue [36 x float] %v181, float 1.0, 35, !dbg !288
  store [36 x float] %v182, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_, align 4, !dbg !288
  br label %bb11, !dbg !289
bb11:
  %v183 = phi i64 [ 0, %bb10 ], [ %v194, %bb14 ]
  %v184 = icmp ult i64 %v183, 36, !dbg !290
  %v185 = xor i1 %v184, 1, !dbg !290
  br i1 %v185, label %bb15, label %bb12, !dbg !290
bb12:
  %v186 = icmp ult i64 %v183, 36, !dbg !291
  br i1 %v186, label %bb13, label %bb18, !dbg !291
bb13:
  br label %bb14, !dbg !291
bb14:
  %v187 = getelementptr inbounds float, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_, i64 %v183, !dbg !291
  %v188 = load float, ptr %v187, align 4, !dbg !291
  %v189 = getelementptr inbounds { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr }, ptr %__cuda_oxide_local_x3209383809627566666572730942756666657273_, i32 0, i32 3, !dbg !292
  %v190 = load ptr, ptr %v189, align 8, !dbg !292
  %v191 = mul i64 %v44, 36, !dbg !293
  %v192 = add i64 %v191, %v183, !dbg !293
  %v193 = getelementptr inbounds float, ptr %v190, i64 %v192, !dbg !243
  store float %v188, ptr %v193, align 4, !dbg !294
  %v194 = add i64 %v183, 1, !dbg !295
  br label %bb11, !dbg !289
bb15:
  ret void, !dbg !296
bb16:
  call void @llvm.trap() #0, !dbg !265
  unreachable, !dbg !265
bb17:
  call void @llvm.trap() #0, !dbg !272
  unreachable, !dbg !272
bb18:
  call void @llvm.trap() #0, !dbg !291
  unreachable, !dbg !291
}

define i1 @sim_env__task__hits_wall(float %v0, float %v1, float %v2, float %v3, float %v4) #0 !dbg !297 {
entry:
  br label %bb0
bb0:
  %v5 = phi float [ %v0, %entry ]
  %v6 = phi float [ %v1, %entry ]
  %v7 = phi float [ %v2, %entry ]
  %v8 = phi float [ %v3, %entry ]
  %v9 = phi float [ %v4, %entry ]
  %v10 = fcmp ogt float %v6, -5.670000076293945, !dbg !300
  %v11 = xor i1 %v10, 1, !dbg !300
  br i1 %v11, label %bb2, label %bb1, !dbg !300
bb1:
  %v12 = fcmp ogt float %v8, -5.670000076293945, !dbg !301
  %v13 = xor i1 %v12, 1, !dbg !301
  br i1 %v13, label %bb2, label %bb4, !dbg !301
bb2:
  %v14 = fcmp olt float %v6, -6.329999923706055, !dbg !302
  %v15 = xor i1 %v14, 1, !dbg !302
  br i1 %v15, label %bb5, label %bb3, !dbg !302
bb3:
  %v16 = fcmp olt float %v8, -6.329999923706055, !dbg !303
  %v17 = xor i1 %v16, 1, !dbg !303
  br i1 %v17, label %bb5, label %bb4, !dbg !303
bb4:
  br label %bb24
bb5:
  %v18 = fsub contract float %v8, %v6, !dbg !304
  %v19 = fcmp une float %v18, 0.0, !dbg !307
  %v20 = xor i1 %v19, 1, !dbg !307
  br i1 %v20, label %bb16, label %bb6, !dbg !307
bb6:
  %v21 = fsub contract float -5.670000076293945, %v6, !dbg !308
  %v22 = fdiv contract float %v21, %v18, !dbg !308
  %v23 = fsub contract float -6.329999923706055, %v6, !dbg !310
  %v24 = fdiv contract float %v23, %v18, !dbg !310
  %v25 = fcmp olt float %v22, %v24, !dbg !312
  %v26 = xor i1 %v25, 1, !dbg !312
  br i1 %v26, label %bb8, label %bb7, !dbg !312
bb7:
  br label %bb9, !dbg !313
bb8:
  br label %bb9, !dbg !313
bb9:
  %v27 = phi float [ %v24, %bb7 ], [ %v22, %bb8 ]
  %v28 = phi float [ %v22, %bb7 ], [ %v24, %bb8 ]
  %v29 = fcmp ogt float %v28, 0.0, !dbg !315
  %v30 = xor i1 %v29, 1, !dbg !315
  br i1 %v30, label %bb11, label %bb10, !dbg !315
bb10:
  br label %bb12, !dbg !316
bb11:
  br label %bb12, !dbg !316
bb12:
  %v31 = phi float [ %v28, %bb10 ], [ 0.0, %bb11 ]
  %v32 = fcmp olt float %v27, 1.0, !dbg !317
  %v33 = xor i1 %v32, 1, !dbg !317
  br i1 %v33, label %bb14, label %bb13, !dbg !317
bb13:
  br label %bb15, !dbg !318
bb14:
  br label %bb15, !dbg !318
bb15:
  %v34 = phi float [ %v27, %bb13 ], [ 1.0, %bb14 ]
  br label %bb16, !dbg !319
bb16:
  %v35 = phi float [ 0.0, %bb5 ], [ %v31, %bb15 ]
  %v36 = phi float [ 1.0, %bb5 ], [ %v34, %bb15 ]
  %v37 = fcmp ogt float %v35, %v36, !dbg !320
  %v38 = xor i1 %v37, 1, !dbg !320
  br i1 %v38, label %bb18, label %bb17, !dbg !320
bb17:
  br label %bb24
bb18:
  %v39 = fsub contract float %v7, %v5, !dbg !321
  %v40 = fmul contract float %v39, %v35, !dbg !321
  %v41 = fadd contract float %v5, %v40, !dbg !322
  %v42 = fsub contract float %v41, %v9, !dbg !322
  %v43 = fmul contract float %v39, %v36, !dbg !324
  %v44 = fadd contract float %v5, %v43, !dbg !325
  %v45 = fsub contract float %v44, %v9, !dbg !325
  %v46 = fcmp olt float %v42, -0.6200000047683716, !dbg !328
  %v47 = xor i1 %v46, 1, !dbg !328
  br i1 %v47, label %bb19, label %bb22, !dbg !328
bb19:
  %v48 = fcmp ogt float %v42, 0.6200000047683716, !dbg !329
  %v49 = xor i1 %v48, 1, !dbg !329
  br i1 %v49, label %bb20, label %bb22, !dbg !329
bb20:
  %v50 = fcmp olt float %v45, -0.6200000047683716, !dbg !330
  %v51 = xor i1 %v50, 1, !dbg !330
  br i1 %v51, label %bb21, label %bb22, !dbg !330
bb21:
  %v52 = fcmp ogt float %v45, 0.6200000047683716, !dbg !331
  br label %bb23, !dbg !328
bb22:
  br label %bb23, !dbg !328
bb23:
  %v53 = phi i1 [ %v52, %bb21 ], [ 1, %bb22 ]
  br label %bb25, !dbg !332
bb24:
  %v54 = phi i1 [ 0, %bb4 ], [ 0, %bb17 ]
  br label %bb25, !dbg !332
bb25:
  %v55 = phi i1 [ %v53, %bb23 ], [ %v54, %bb24 ]
  ret i1 %v55, !dbg !332
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env(ptr %v0) alwaysinline #0 !dbg !333 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1, !dbg !334
  %v3 = xor i1 %v2, 1, !dbg !334
  br i1 %v3, label %bb2, label %bb1, !dbg !334
bb1:
  br label %bb8, !dbg !335
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0, !dbg !336
  br label %bb3, !dbg !336
bb3:
  %v5 = icmp eq i32 %v4, 1, !dbg !336
  br i1 %v5, label %bb4, label %bb5, !dbg !336
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0, !dbg !337
  br label %bb6, !dbg !337
bb5:
  br label %bb7, !dbg !336
bb6:
  %v7 = icmp eq i32 %v6, 1, !dbg !337
  br label %bb7, !dbg !336
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8, !dbg !335
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1, !dbg !339
  br i1 %v10, label %bb9, label %bb10, !dbg !339
bb9:
  %v11 = icmp eq i8 0, 2, !dbg !340
  %v12 = xor i1 %v11, 1, !dbg !340
  br i1 %v12, label %bb11, label %bb10, !dbg !340
bb10:
  br label %bb17, !dbg !341
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0, !dbg !342
  br label %bb12, !dbg !342
bb12:
  %v14 = icmp eq i32 %v13, 1, !dbg !342
  br i1 %v14, label %bb13, label %bb14, !dbg !342
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0, !dbg !343
  br label %bb15, !dbg !343
bb14:
  br label %bb16, !dbg !342
bb15:
  %v16 = icmp eq i32 %v15, 1, !dbg !343
  br label %bb16, !dbg !342
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17, !dbg !341
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1, !dbg !345
  br i1 %v19, label %bb19, label %bb18, !dbg !345
bb18:
  br label %bb20, !dbg !345
bb19:
  br label %bb20, !dbg !345
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20, !dbg !346
}


@llvm.used = appending global [3 x ptr] [ptr @control, ptr @finish, ptr @reset], section "llvm.metadata"

attributes #0 = { convergent }

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!347, !348}
!347 = !{i32 2, !"Dwarf Version", i32 2}
!348 = !{i32 2, !"Debug Info Version", i32 3}
!0 = !DIFile(filename: "kernels.rs", directory: "/home/jaren/visualencoder/sim/crates/sim-env/src")
!1 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !0, producer: "cuda-oxide", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DISubroutineType(types: !{null})
!3 = distinct !DISubprogram(name: "control", linkageName: "control", scope: !0, file: !0, line: 131, type: !2, scopeLine: 131, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!4 = !DILexicalBlock(scope: !3, file: !0, line: 131, column: 82)
!5 = !DILocation(line: 132, column: 17, scope: !4)
!6 = !DILexicalBlock(scope: !4, file: !0, line: 132, column: 9)
!7 = !DILocation(line: 133, column: 17, scope: !6)
!8 = !DILocation(line: 133, column: 12, scope: !6)
!9 = !DILocation(line: 150, column: 6, scope: !3)
!10 = !DILocation(line: 134, column: 30, scope: !6)
!11 = !DILexicalBlock(scope: !6, file: !0, line: 134, column: 9)
!12 = !DILexicalBlock(scope: !11, file: !0, line: 135, column: 9)
!13 = !DILexicalBlock(scope: !12, file: !0, line: 136, column: 9)
!14 = !DILexicalBlock(scope: !13, file: !0, line: 137, column: 9)
!15 = !DIFile(filename: "mut_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!16 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut sim_state::BodyAction>::add", scope: !15, file: !15, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!17 = !DILocation(line: 143, column: 29, scope: !14)
!18 = !DILocation(line: 970, column: 18, scope: !16, inlinedAt: !17)
!19 = !DILocation(line: 134, column: 29, scope: !6)
!20 = !DIFile(filename: "const_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!21 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const i64>::add", scope: !20, file: !20, line: 837, type: !2, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!22 = !DILocation(line: 135, column: 40, scope: !11)
!23 = !DILocation(line: 871, column: 18, scope: !21, inlinedAt: !22)
!24 = !DILocation(line: 135, column: 31, scope: !11)
!25 = !DILocation(line: 136, column: 23, scope: !12)
!26 = !DILocation(line: 136, column: 22, scope: !12)
!27 = !DILocation(line: 137, column: 34, scope: !13)
!28 = !DILocation(line: 137, column: 33, scope: !13)
!29 = !DILocation(line: 138, column: 22, scope: !14)
!30 = !DILocation(line: 138, column: 9, scope: !14)
!31 = !DILocation(line: 139, column: 22, scope: !14)
!32 = !DILocation(line: 139, column: 9, scope: !14)
!33 = !DILocation(line: 140, column: 9, scope: !14)
!34 = !DILocation(line: 142, column: 13, scope: !14)
!35 = !DILocation(line: 144, column: 41, scope: !14)
!36 = !DILocation(line: 144, column: 31, scope: !14)
!37 = !DILocation(line: 144, column: 25, scope: !14)
!38 = !DILocation(line: 145, column: 42, scope: !14)
!39 = !DILocation(line: 145, column: 35, scope: !14)
!40 = !DILocation(line: 145, column: 28, scope: !14)
!41 = !DILocation(line: 145, column: 21, scope: !14)
!42 = !DILocation(line: 145, column: 68, scope: !14)
!43 = !DILocation(line: 145, column: 62, scope: !14)
!44 = !DILocation(line: 146, column: 35, scope: !14)
!45 = !DILocation(line: 146, column: 27, scope: !14)
!46 = !DILocation(line: 146, column: 21, scope: !14)
!47 = !DILocation(line: 144, column: 24, scope: !14)
!48 = !DILocation(line: 147, column: 25, scope: !14)
!49 = !DILocation(line: 143, column: 38, scope: !14)
!50 = !DILocation(line: 143, column: 14, scope: !14)
!51 = !DILocation(line: 143, column: 13, scope: !14)
!52 = distinct !DISubprogram(name: "reset", linkageName: "reset", scope: !0, file: !0, line: 115, type: !2, scopeLine: 115, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!53 = !DILexicalBlock(scope: !52, file: !0, line: 115, column: 59)
!54 = !DILocation(line: 116, column: 17, scope: !53)
!55 = !DILexicalBlock(scope: !53, file: !0, line: 116, column: 9)
!56 = !DILocation(line: 117, column: 17, scope: !55)
!57 = !DILocation(line: 117, column: 12, scope: !55)
!58 = !DILocation(line: 128, column: 6, scope: !52)
!59 = !DILocation(line: 119, column: 14, scope: !55)
!60 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut f32>::add", scope: !15, file: !15, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!61 = !DILocation(line: 124, column: 32, scope: !55)
!62 = !DILocation(line: 970, column: 18, scope: !60, inlinedAt: !61)
!63 = !DILocation(line: 119, column: 13, scope: !55)
!64 = !DILocation(line: 120, column: 58, scope: !55)
!65 = !DILocation(line: 120, column: 38, scope: !55)
!66 = !DILocation(line: 120, column: 14, scope: !55)
!67 = !DILocation(line: 120, column: 13, scope: !55)
!68 = !DILocation(line: 121, column: 14, scope: !55)
!69 = !DILocation(line: 121, column: 13, scope: !55)
!70 = !DILocation(line: 122, column: 14, scope: !55)
!71 = !DILocation(line: 122, column: 13, scope: !55)
!72 = !DILocation(line: 123, column: 14, scope: !55)
!73 = !DILocation(line: 123, column: 13, scope: !55)
!74 = !DILocation(line: 124, column: 14, scope: !55)
!75 = !DILocation(line: 124, column: 13, scope: !55)
!76 = !DILocation(line: 125, column: 13, scope: !55)
!77 = !DILocation(line: 126, column: 13, scope: !55)
!78 = distinct !DISubprogram(name: "finish", linkageName: "finish", scope: !0, file: !0, line: 153, type: !2, scopeLine: 153, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!79 = !DILexicalBlock(scope: !78, file: !0, line: 153, column: 60)
!80 = !DILocation(line: 154, column: 17, scope: !79)
!81 = !DILexicalBlock(scope: !79, file: !0, line: 154, column: 9)
!82 = !DILocation(line: 155, column: 17, scope: !81)
!83 = !DILocation(line: 155, column: 12, scope: !81)
!84 = !DILocation(line: 182, column: 6, scope: !78)
!85 = !DILocation(line: 156, column: 30, scope: !81)
!86 = !DILexicalBlock(scope: !81, file: !0, line: 156, column: 9)
!87 = !DILexicalBlock(scope: !86, file: !0, line: 157, column: 9)
!88 = !DILexicalBlock(scope: !87, file: !0, line: 158, column: 9)
!89 = !DILexicalBlock(scope: !88, file: !0, line: 159, column: 9)
!90 = !DILexicalBlock(scope: !89, file: !0, line: 162, column: 9)
!91 = !DILexicalBlock(scope: !90, file: !0, line: 164, column: 9)
!92 = !DILexicalBlock(scope: !91, file: !0, line: 165, column: 9)
!93 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut u64>::add", scope: !15, file: !15, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!94 = !DILocation(line: 174, column: 50, scope: !92)
!95 = !DILocation(line: 970, column: 18, scope: !93, inlinedAt: !94)
!96 = !DILocation(line: 156, column: 29, scope: !81)
!97 = !DILocation(line: 157, column: 34, scope: !86)
!98 = !DILocation(line: 157, column: 33, scope: !86)
!99 = !DILocation(line: 158, column: 23, scope: !87)
!100 = !DILocation(line: 158, column: 42, scope: !87)
!101 = !DILocation(line: 159, column: 25, scope: !88)
!102 = !DILocation(line: 159, column: 58, scope: !88)
!103 = !DILocation(line: 160, column: 26, scope: !88)
!104 = !DILocation(line: 160, column: 56, scope: !88)
!105 = !DILocation(line: 161, column: 26, scope: !88)
!106 = !DILocation(line: 161, column: 44, scope: !88)
!107 = !DILocation(line: 160, column: 16, scope: !88)
!108 = !DILocation(line: 162, column: 24, scope: !89)
!109 = !DILocation(line: 162, column: 37, scope: !89)
!110 = !DILocation(line: 162, column: 23, scope: !89)
!111 = !DILocation(line: 163, column: 9, scope: !90)
!112 = !DILocation(line: 164, column: 20, scope: !90)
!113 = !DILocation(line: 164, column: 33, scope: !90)
!114 = !DILocation(line: 164, column: 44, scope: !90)
!115 = !DILocation(line: 165, column: 23, scope: !91)
!116 = !DILocation(line: 165, column: 36, scope: !91)
!117 = !DILocation(line: 165, column: 22, scope: !91)
!118 = !DILocation(line: 166, column: 18, scope: !91)
!119 = !DILocation(line: 166, column: 15, scope: !91)
!120 = !DILocation(line: 166, column: 42, scope: !91)
!121 = !DILocation(line: 166, column: 39, scope: !91)
!122 = !DILocation(line: 168, column: 14, scope: !92)
!123 = !DILocation(line: 168, column: 13, scope: !92)
!124 = !DILocation(line: 169, column: 39, scope: !92)
!125 = !DILocation(line: 169, column: 36, scope: !92)
!126 = !DILocation(line: 169, column: 14, scope: !92)
!127 = !DILocation(line: 169, column: 13, scope: !92)
!128 = !DILocation(line: 170, column: 42, scope: !92)
!129 = !DILocation(line: 170, column: 39, scope: !92)
!130 = !DILocation(line: 170, column: 14, scope: !92)
!131 = !DILocation(line: 170, column: 13, scope: !92)
!132 = !DILocation(line: 171, column: 44, scope: !92)
!133 = !DILocation(line: 171, column: 41, scope: !92)
!134 = !DILocation(line: 171, column: 14, scope: !92)
!135 = !DILocation(line: 171, column: 13, scope: !92)
!136 = !DILocation(line: 172, column: 13, scope: !92)
!137 = !DILocation(line: 173, column: 16, scope: !92)
!138 = !DILocation(line: 174, column: 33, scope: !92)
!139 = !DILocation(line: 174, column: 31, scope: !92)
!140 = !DIFile(filename: "mod.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/num")
!141 = distinct !DISubprogram(name: "core::num::<impl u64>::wrapping_add", scope: !140, file: !140, line: 1322, type: !2, scopeLine: 1322, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!142 = !DILocation(line: 174, column: 58, scope: !92)
!143 = !DIFile(filename: "uint_macros.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/num")
!144 = !DILexicalBlockFile(scope: !141, file: !143, discriminator: 0)
!145 = !DILocation(line: 2652, column: 13, scope: !144, inlinedAt: !142)
!146 = !DILexicalBlock(scope: !92, file: !0, line: 174, column: 17)
!147 = !DILocation(line: 175, column: 17, scope: !146)
!148 = !DILocation(line: 176, column: 17, scope: !146)
!149 = !DILocation(line: 173, column: 13, scope: !92)
!150 = !DILocation(line: 180, column: 13, scope: !92)
!151 = !DIFile(filename: "thread.rs", directory: "/home/jaren/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/348895e/cuda-oxide/crates/cuda-device/src")
!152 = distinct !DISubprogram(name: "cuda_device::__internal::index_1d::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env", scope: !151, file: !151, line: 648, type: !2, scopeLine: 648, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!153 = !DILocation(line: 652, column: 13, scope: !152)
!154 = !DILocation(line: 653, column: 13, scope: !152)
!155 = !DILocation(line: 654, column: 13, scope: !152)
!156 = !DILexicalBlock(scope: !152, file: !151, line: 651, column: 9)
!157 = !DILocation(line: 656, column: 21, scope: !156)
!158 = !DILocation(line: 656, column: 54, scope: !156)
!159 = !DILexicalBlock(scope: !156, file: !151, line: 656, column: 9)
!160 = distinct !DISubprogram(name: "cuda_device::ThreadIndex::<'_>::new::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", scope: !151, file: !151, line: 303, type: !2, scopeLine: 303, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!161 = !DILocation(line: 657, column: 18, scope: !159)
!162 = !DILocation(line: 312, column: 21, scope: !160, inlinedAt: !161)
!163 = distinct !DISubprogram(name: "cuda_device::__internal::checked_axis", scope: !151, file: !151, line: 601, type: !2, scopeLine: 601, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!164 = !DILocation(line: 651, column: 19, scope: !152)
!165 = !DILocation(line: 602, column: 21, scope: !163, inlinedAt: !164)
!166 = !DILocation(line: 603, column: 26, scope: !163, inlinedAt: !164)
!167 = !DILocation(line: 604, column: 22, scope: !163, inlinedAt: !164)
!168 = !DILocation(line: 605, column: 12, scope: !163, inlinedAt: !164)
!169 = !DILocation(line: 605, column: 39, scope: !163, inlinedAt: !164)
!170 = !DILocation(line: 605, column: 31, scope: !163, inlinedAt: !164)
!171 = !DILocation(line: 605, column: 74, scope: !163, inlinedAt: !164)
!172 = !DILocation(line: 605, column: 9, scope: !163, inlinedAt: !164)
!173 = !DILocation(line: 608, column: 13, scope: !163, inlinedAt: !164)
!174 = !DILocation(line: 312, column: 30, scope: !160, inlinedAt: !161)
!175 = !DILocation(line: 312, column: 18, scope: !160, inlinedAt: !161)
!176 = !DILocation(line: 658, column: 6, scope: !152)
!177 = distinct !DISubprogram(name: "sim_env::kernels::device::reset_one", linkageName: "sim_env__kernels__device__reset_one", scope: !0, file: !0, line: 51, type: !2, scopeLine: 51, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!178 = !DILocation(line: 52, column: 20, scope: !177)
!179 = !DILexicalBlock(scope: !177, file: !0, line: 52, column: 9)
!180 = !DILocation(line: 53, column: 31, scope: !179)
!181 = !DILocation(line: 53, column: 44, scope: !179)
!182 = !DILocation(line: 53, column: 63, scope: !179)
!183 = !DIFile(filename: "task.rs", directory: "/home/jaren/visualencoder/sim/crates/sim-env/src")
!184 = distinct !DISubprogram(name: "task::episode_gap", scope: !183, file: !183, line: 8, type: !2, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!185 = !DILocation(line: 53, column: 19, scope: !179)
!186 = !DILocation(line: 9, column: 24, scope: !184, inlinedAt: !185)
!187 = distinct !DISubprogram(name: "core::num::<impl u64>::wrapping_mul", scope: !140, file: !140, line: 1322, type: !2, scopeLine: 1322, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!188 = !DILocation(line: 13, column: 25, scope: !184, inlinedAt: !185)
!189 = !DILexicalBlockFile(scope: !187, file: !143, discriminator: 0)
!190 = !DILocation(line: 2728, column: 13, scope: !189, inlinedAt: !188)
!191 = !DILocation(line: 9, column: 17, scope: !184, inlinedAt: !185)
!192 = !DILocation(line: 11, column: 11, scope: !184, inlinedAt: !185)
!193 = !DILocation(line: 12, column: 14, scope: !184, inlinedAt: !185)
!194 = !DILocation(line: 12, column: 9, scope: !184, inlinedAt: !185)
!195 = !DILocation(line: 13, column: 14, scope: !184, inlinedAt: !185)
!196 = !DILocation(line: 13, column: 9, scope: !184, inlinedAt: !185)
!197 = !DILocation(line: 14, column: 10, scope: !184, inlinedAt: !185)
!198 = !DILocation(line: 14, column: 5, scope: !184, inlinedAt: !185)
!199 = !DILocation(line: 15, column: 5, scope: !184, inlinedAt: !185)
!200 = !DILexicalBlock(scope: !179, file: !0, line: 53, column: 9)
!201 = !DILocation(line: 56, column: 27, scope: !200)
!202 = !DILocation(line: 57, column: 27, scope: !200)
!203 = !DILocation(line: 58, column: 27, scope: !200)
!204 = !DILocation(line: 59, column: 35, scope: !200)
!205 = !DILocation(line: 55, column: 38, scope: !200)
!206 = !DILocation(line: 55, column: 14, scope: !200)
!207 = !DILexicalBlock(scope: !200, file: !0, line: 64, column: 13)
!208 = !DILexicalBlock(scope: !207, file: !0, line: 69, column: 13)
!209 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut sim_state::BodyState>::add", scope: !15, file: !15, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!210 = !DILocation(line: 71, column: 29, scope: !208)
!211 = !DILocation(line: 970, column: 18, scope: !209, inlinedAt: !210)
!212 = !DILocation(line: 55, column: 13, scope: !200)
!213 = !DILocation(line: 61, column: 36, scope: !200)
!214 = !DILocation(line: 61, column: 14, scope: !200)
!215 = !DILocation(line: 61, column: 13, scope: !200)
!216 = !DILocation(line: 65, column: 28, scope: !200)
!217 = !DILocation(line: 65, column: 27, scope: !200)
!218 = !DILocation(line: 64, column: 24, scope: !200)
!219 = !DILocation(line: 68, column: 33, scope: !207)
!220 = !DILocation(line: 68, column: 53, scope: !207)
!221 = !DILocation(line: 68, column: 13, scope: !207)
!222 = !DILocation(line: 70, column: 33, scope: !208)
!223 = !DILocation(line: 70, column: 13, scope: !208)
!224 = !DILocation(line: 71, column: 33, scope: !208)
!225 = !DILocation(line: 71, column: 13, scope: !208)
!226 = !DILocation(line: 73, column: 6, scope: !177)
!227 = distinct !DISubprogram(name: "sim_env::kernels::device::observe_one", linkageName: "sim_env__kernels__device__observe_one", scope: !0, file: !0, line: 75, type: !2, scopeLine: 75, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!228 = !DILocation(line: 76, column: 30, scope: !227)
!229 = !DILexicalBlock(scope: !227, file: !0, line: 76, column: 9)
!230 = !DILexicalBlock(scope: !229, file: !0, line: 77, column: 9)
!231 = !DILexicalBlock(scope: !230, file: !0, line: 78, column: 9)
!232 = !DILexicalBlock(scope: !231, file: !0, line: 79, column: 9)
!233 = !DILexicalBlock(scope: !232, file: !0, line: 80, column: 9)
!234 = !DILexicalBlock(scope: !233, file: !0, line: 81, column: 9)
!235 = !DILexicalBlock(scope: !234, file: !0, line: 83, column: 9)
!236 = !DILexicalBlock(scope: !235, file: !0, line: 84, column: 9)
!237 = !DILexicalBlock(scope: !236, file: !0, line: 96, column: 9)
!238 = !DILexicalBlock(scope: !237, file: !0, line: 97, column: 9)
!239 = !DILexicalBlock(scope: !238, file: !0, line: 98, column: 9)
!240 = !DILexicalBlock(scope: !239, file: !0, line: 99, column: 9)
!241 = !DILexicalBlock(scope: !240, file: !0, line: 100, column: 9)
!242 = !DILocation(line: 109, column: 39, scope: !241)
!243 = !DILocation(line: 970, column: 18, scope: !60, inlinedAt: !242)
!244 = !DILocation(line: 76, column: 29, scope: !227)
!245 = !DILocation(line: 77, column: 30, scope: !229)
!246 = !DILocation(line: 77, column: 29, scope: !229)
!247 = !DILocation(line: 78, column: 14, scope: !230)
!248 = !DILocation(line: 78, column: 17, scope: !230)
!249 = !DILocation(line: 78, column: 20, scope: !230)
!250 = !DILocation(line: 79, column: 20, scope: !231)
!251 = !DILocation(line: 80, column: 21, scope: !232)
!252 = !DILocation(line: 81, column: 24, scope: !233)
!253 = !DILocation(line: 81, column: 34, scope: !233)
!254 = !DILocation(line: 81, column: 43, scope: !233)
!255 = !DILocation(line: 82, column: 13, scope: !233)
!256 = !DILocation(line: 82, column: 31, scope: !233)
!257 = !DILocation(line: 82, column: 49, scope: !233)
!258 = !DILocation(line: 82, column: 67, scope: !233)
!259 = !DILocation(line: 81, column: 23, scope: !233)
!260 = !DILocation(line: 83, column: 27, scope: !234)
!261 = !DILocation(line: 83, column: 48, scope: !234)
!262 = !DILocation(line: 83, column: 26, scope: !234)
!263 = !DILocation(line: 85, column: 9, scope: !236)
!264 = !DILocation(line: 85, column: 15, scope: !236)
!265 = !DILocation(line: 86, column: 56, scope: !236)
!266 = !DILocation(line: 86, column: 23, scope: !236)
!267 = !DILocation(line: 86, column: 43, scope: !236)
!268 = !DILocation(line: 86, column: 22, scope: !236)
!269 = !DILocation(line: 87, column: 13, scope: !236)
!270 = !DILocation(line: 90, column: 9, scope: !236)
!271 = !DILocation(line: 90, column: 15, scope: !236)
!272 = !DILocation(line: 91, column: 59, scope: !236)
!273 = !DILocation(line: 91, column: 23, scope: !236)
!274 = !DILocation(line: 91, column: 46, scope: !236)
!275 = !DILocation(line: 91, column: 22, scope: !236)
!276 = !DILocation(line: 92, column: 13, scope: !236)
!277 = !DILocation(line: 96, column: 17, scope: !236)
!278 = !DILocation(line: 97, column: 17, scope: !237)
!279 = !DILocation(line: 98, column: 17, scope: !238)
!280 = !DILocation(line: 99, column: 17, scope: !239)
!281 = !DILocation(line: 102, column: 33, scope: !240)
!282 = !DILocation(line: 102, column: 37, scope: !240)
!283 = !DILocation(line: 102, column: 41, scope: !240)
!284 = !DILocation(line: 104, column: 32, scope: !240)
!285 = !DILocation(line: 104, column: 40, scope: !240)
!286 = !DILocation(line: 104, column: 52, scope: !240)
!287 = !DILocation(line: 104, column: 48, scope: !240)
!288 = !DILocation(line: 100, column: 22, scope: !240)
!289 = !DILocation(line: 108, column: 9, scope: !241)
!290 = !DILocation(line: 108, column: 15, scope: !241)
!291 = !DILocation(line: 109, column: 57, scope: !241)
!292 = !DILocation(line: 109, column: 23, scope: !241)
!293 = !DILocation(line: 109, column: 43, scope: !241)
!294 = !DILocation(line: 109, column: 22, scope: !241)
!295 = !DILocation(line: 110, column: 13, scope: !241)
!296 = !DILocation(line: 112, column: 6, scope: !227)
!297 = distinct !DISubprogram(name: "sim_env::task::hits_wall", linkageName: "sim_env__task__hits_wall", scope: !183, file: !183, line: 19, type: !2, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!298 = !DILexicalBlock(scope: !297, file: !183, line: 20, column: 5)
!299 = !DILexicalBlock(scope: !298, file: !183, line: 21, column: 5)
!300 = !DILocation(line: 22, column: 9, scope: !299)
!301 = !DILocation(line: 22, column: 26, scope: !299)
!302 = !DILocation(line: 22, column: 41, scope: !299)
!303 = !DILocation(line: 22, column: 57, scope: !299)
!304 = !DILocation(line: 25, column: 14, scope: !299)
!305 = !DILexicalBlock(scope: !299, file: !183, line: 25, column: 5)
!306 = !DILexicalBlock(scope: !305, file: !183, line: 26, column: 5)
!307 = !DILocation(line: 27, column: 8, scope: !306)
!308 = !DILocation(line: 28, column: 18, scope: !306)
!309 = !DILexicalBlock(scope: !306, file: !183, line: 28, column: 9)
!310 = !DILocation(line: 29, column: 18, scope: !309)
!311 = !DILexicalBlock(scope: !309, file: !183, line: 29, column: 9)
!312 = !DILocation(line: 30, column: 27, scope: !311)
!313 = !DILocation(line: 30, column: 24, scope: !311)
!314 = !DILexicalBlock(scope: !311, file: !183, line: 30, column: 9)
!315 = !DILocation(line: 31, column: 12, scope: !314)
!316 = !DILocation(line: 31, column: 9, scope: !314)
!317 = !DILocation(line: 32, column: 12, scope: !314)
!318 = !DILocation(line: 32, column: 9, scope: !314)
!319 = !DILocation(line: 27, column: 5, scope: !306)
!320 = !DILocation(line: 34, column: 8, scope: !306)
!321 = !DILocation(line: 35, column: 22, scope: !306)
!322 = !DILocation(line: 35, column: 14, scope: !306)
!323 = !DILexicalBlock(scope: !306, file: !183, line: 35, column: 5)
!324 = !DILocation(line: 36, column: 22, scope: !323)
!325 = !DILocation(line: 36, column: 14, scope: !323)
!326 = !DILexicalBlock(scope: !323, file: !183, line: 36, column: 5)
!327 = !DILexicalBlock(scope: !326, file: !183, line: 37, column: 5)
!328 = !DILocation(line: 38, column: 5, scope: !327)
!329 = !DILocation(line: 38, column: 24, scope: !327)
!330 = !DILocation(line: 38, column: 42, scope: !327)
!331 = !DILocation(line: 38, column: 61, scope: !327)
!332 = !DILocation(line: 39, column: 2, scope: !297)
!333 = distinct !DISubprogram(name: "cuda_device::__internal::one_dimensional_launch::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env", scope: !151, file: !151, line: 613, type: !2, scopeLine: 613, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!334 = !DILocation(line: 616, column: 28, scope: !333)
!335 = !DILocation(line: 616, column: 25, scope: !333)
!336 = !DILocation(line: 619, column: 13, scope: !333)
!337 = !DILocation(line: 619, column: 41, scope: !333)
!338 = !DILexicalBlock(scope: !333, file: !151, line: 616, column: 9)
!339 = !DILocation(line: 621, column: 28, scope: !338)
!340 = !DILocation(line: 621, column: 59, scope: !338)
!341 = !DILocation(line: 621, column: 25, scope: !338)
!342 = !DILocation(line: 624, column: 13, scope: !338)
!343 = !DILocation(line: 624, column: 41, scope: !338)
!344 = !DILexicalBlock(scope: !338, file: !151, line: 621, column: 9)
!345 = !DILocation(line: 626, column: 9, scope: !344)
!346 = !DILocation(line: 627, column: 6, scope: !333)
