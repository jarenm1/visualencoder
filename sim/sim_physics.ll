; ModuleID = 'builtin.module'
source_filename = "sim_physics"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

; shared source: sim_physics::kernels::device::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_reduce_images::MAXIMUM
@__shared_mem_f6621971da3d7ca3_01b97a1b1cfda569 = addrspace(3) global [256 x float] zeroinitializer, align 4
; shared source: sim_physics::kernels::device::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_reduce_images::MINIMUM
@__shared_mem_f6621971da3d7ca3_880cddad8d2c4a2f = addrspace(3) global [256 x float] zeroinitializer, align 4
; shared source: sim_physics::kernels::device::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_reduce_images::BLUE
@__shared_mem_f6621971da3d7ca3_6b287ec6666962c3 = addrspace(3) global [256 x float] zeroinitializer, align 4
; shared source: sim_physics::kernels::device::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_reduce_images::GREEN
@__shared_mem_f6621971da3d7ca3_e2553c23b4905f70 = addrspace(3) global [256 x float] zeroinitializer, align 4
; shared source: sim_physics::kernels::device::cuda_oxide_codegen_v1_cuda_oxide_kernel_246e25db_reduce_images::RED
@__shared_mem_f6621971da3d7ca3_b194c80a4296d25c = addrspace(3) global [256 x float] zeroinitializer, align 4
declare void @llvm.trap()

define ptx_kernel void @rgba8_to_nchw(ptr %v0, ptr %v1, i32 %v2, i32 %v3) #0 !dbg !3 {
entry:
  %v4 = alloca {  }, align 1
  br label %bb0
bb0:
  %v5 = phi ptr [ %v0, %entry ]
  %v6 = phi ptr [ %v1, %entry ]
  %v7 = phi i32 [ %v2, %entry ]
  %v8 = phi i32 [ %v3, %entry ]
  %v9 = call i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics(ptr %v4) #0, !dbg !5
  br label %bb1, !dbg !5
bb1:
  %v10 = zext i32 %v7 to i64, !dbg !7
  %v11 = icmp uge i64 %v9, %v10, !dbg !8
  %v12 = xor i1 %v11, 1, !dbg !8
  br i1 %v12, label %bb3, label %bb2, !dbg !8
bb2:
  br label %bb8, !dbg !9
bb3:
  %v13 = zext i32 %v8 to i64, !dbg !10
  %v14 = icmp eq i64 %v13, 0, !dbg !12
  %v15 = xor i1 %v14, 1, !dbg !12
  br i1 %v15, label %bb4, label %bb9, !dbg !12
bb4:
  br label %bb5, !dbg !12
bb5:
  %v16 = udiv i64 %v9, %v13, !dbg !12
  %v17 = xor i1 %v14, 1, !dbg !14
  br i1 %v17, label %bb6, label %bb10, !dbg !14
bb6:
  br label %bb7, !dbg !14
bb7:
  %v18 = urem i64 %v9, %v13, !dbg !14
  %v19 = mul i64 %v16, 3, !dbg !16
  %v20 = mul i64 %v19, %v13, !dbg !16
  %v21 = add i64 %v20, %v18, !dbg !16
  %v22 = mul i64 %v9, 4, !dbg !19
  %v23 = getelementptr inbounds i8, ptr %v5, i64 %v22, !dbg !23
  %v24 = load i8, ptr %v23, align 1, !dbg !24
  %v25 = uitofp i8 %v24 to float, !dbg !24
  %v26 = getelementptr inbounds float, ptr %v6, i64 %v21, !dbg !28
  %v27 = fmul contract float %v25, 0.003921568859368563, !dbg !29
  store float %v27, ptr %v26, align 4, !dbg !29
  %v28 = add i64 %v22, 1, !dbg !30
  %v29 = getelementptr inbounds i8, ptr %v5, i64 %v28, !dbg !23
  %v30 = load i8, ptr %v29, align 1, !dbg !31
  %v31 = uitofp i8 %v30 to float, !dbg !31
  %v32 = add i64 %v21, %v13, !dbg !32
  %v33 = getelementptr inbounds float, ptr %v6, i64 %v32, !dbg !28
  %v34 = fmul contract float %v31, 0.003921568859368563, !dbg !33
  store float %v34, ptr %v33, align 4, !dbg !33
  %v35 = add i64 %v22, 2, !dbg !34
  %v36 = getelementptr inbounds i8, ptr %v5, i64 %v35, !dbg !23
  %v37 = load i8, ptr %v36, align 1, !dbg !35
  %v38 = uitofp i8 %v37 to float, !dbg !35
  %v39 = mul i64 2, %v13, !dbg !36
  %v40 = add i64 %v21, %v39, !dbg !37
  %v41 = getelementptr inbounds float, ptr %v6, i64 %v40, !dbg !28
  %v42 = fmul contract float %v38, 0.003921568859368563, !dbg !38
  store float %v42, ptr %v41, align 4, !dbg !38
  br label %bb8, !dbg !9
bb8:
  ret void, !dbg !9
bb9:
  call void @llvm.trap() #0, !dbg !12
  unreachable, !dbg !12
bb10:
  call void @llvm.trap() #0, !dbg !14
  unreachable, !dbg !14
}

define ptx_kernel void @integrate(ptr %v0, ptr %v1, i32 %v2, { [4 x float], float, float, float, float, float, [3 x float] } %v3) #0 !dbg !39 {
entry:
  %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_ = alloca { [4 x float], float, float, float, float, float, [3 x float] }, align 16
  %v5 = alloca {  }, align 1
  %__cuda_oxide_local_x313109363409737461746509426f64795374617465_ = alloca { [4 x float], [4 x float], [4 x float], [4 x float] }, align 16
  %__cuda_oxide_local_x31303409313609616374696f6e095b6633323b20345d_ = alloca [4 x float], align 4
  %__cuda_oxide_local_x31303509313609616374696f6e095b6633323b20345d_ = alloca [4 x float], align 4
  %__cuda_oxide_local_x3130370931360967726176697479095b6633323b20345d_ = alloca [4 x float], align 4
  br label %bb0
bb0:
  %v10 = phi ptr [ %v0, %entry ]
  %v11 = phi ptr [ %v1, %entry ]
  %v12 = phi i32 [ %v2, %entry ]
  %v13 = phi { [4 x float], float, float, float, float, float, [3 x float] } [ %v3, %entry ]
  store { [4 x float], float, float, float, float, float, [3 x float] } %v13, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, align 16
  %v14 = call i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics(ptr %v5) #0, !dbg !41
  br label %bb1, !dbg !41
bb1:
  %v15 = zext i32 %v12 to i64, !dbg !43
  %v16 = icmp uge i64 %v14, %v15, !dbg !44
  %v17 = xor i1 %v16, 1, !dbg !44
  br i1 %v17, label %bb3, label %bb2, !dbg !44
bb2:
  br label %bb26, !dbg !45
bb3:
  %v18 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v10, i64 %v14, !dbg !48
  %v19 = load { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %v18, align 16, !dbg !49
  store { [4 x float], [4 x float], [4 x float], [4 x float] } %v19, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, align 16, !dbg !49
  %v20 = getelementptr inbounds { [4 x float], [4 x float] }, ptr %v11, i64 %v14, !dbg !53
  %v21 = getelementptr inbounds { [4 x float], [4 x float] }, ptr %v20, i32 0, i32 0, !dbg !54
  %v22 = load [4 x float], ptr %v21, align 16, !dbg !54
  store [4 x float] %v22, ptr %__cuda_oxide_local_x31303409313609616374696f6e095b6633323b20345d_, align 4, !dbg !54
  %v23 = getelementptr inbounds { [4 x float], [4 x float] }, ptr %v20, i32 0, i32 1, !dbg !54
  %v24 = load [4 x float], ptr %v23, align 16, !dbg !54
  store [4 x float] %v24, ptr %__cuda_oxide_local_x31303509313609616374696f6e095b6633323b20345d_, align 4, !dbg !54
  %v25 = getelementptr inbounds { [4 x float], float, float, float, float, float, [3 x float] }, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, i32 0, i32 1, !dbg !56
  %v26 = load float, ptr %v25, align 16, !dbg !56
  %v27 = getelementptr inbounds { [4 x float], float, float, float, float, float, [3 x float] }, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, i32 0, i32 0, !dbg !58
  %v28 = load [4 x float], ptr %v27, align 16, !dbg !58
  store [4 x float] %v28, ptr %__cuda_oxide_local_x3130370931360967726176697479095b6633323b20345d_, align 4, !dbg !58
  %v29 = getelementptr inbounds { [4 x float], float, float, float, float, float, [3 x float] }, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, i32 0, i32 4, !dbg !60
  %v30 = load float, ptr %v29, align 4, !dbg !60
  %v31 = fmul contract float %v30, %v26, !dbg !60
  %v32 = fadd contract float 1.0, %v31, !dbg !61
  %v33 = fdiv contract float 1.0, %v32, !dbg !62
  %v34 = getelementptr inbounds { [4 x float], float, float, float, float, float, [3 x float] }, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, i32 0, i32 5, !dbg !64
  %v35 = load float, ptr %v34, align 16, !dbg !64
  %v36 = fmul contract float %v35, %v26, !dbg !64
  %v37 = fadd contract float 1.0, %v36, !dbg !65
  %v38 = fdiv contract float 1.0, %v37, !dbg !66
  br label %bb4, !dbg !69
bb4:
  %v39 = phi i64 [ 0, %bb3 ], [ %v90, %bb23 ]
  %v40 = icmp ult i64 %v39, 3, !dbg !70
  %v41 = xor i1 %v40, 1, !dbg !70
  br i1 %v41, label %bb24, label %bb5, !dbg !70
bb5:
  %v42 = icmp ult i64 %v39, 4, !dbg !71
  br i1 %v42, label %bb6, label %bb27, !dbg !71
bb6:
  br label %bb7, !dbg !71
bb7:
  %v43 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 1, !dbg !71
  %v44 = getelementptr inbounds float, ptr %v43, i64 %v39, !dbg !71
  %v45 = load float, ptr %v44, align 4, !dbg !71
  %v46 = icmp ult i64 %v39, 4, !dbg !72
  br i1 %v46, label %bb8, label %bb28, !dbg !72
bb8:
  br label %bb9, !dbg !72
bb9:
  %v47 = getelementptr inbounds float, ptr %__cuda_oxide_local_x31303409313609616374696f6e095b6633323b20345d_, i64 %v39, !dbg !72
  %v48 = load float, ptr %v47, align 4, !dbg !72
  %v49 = getelementptr inbounds { [4 x float], float, float, float, float, float, [3 x float] }, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, i32 0, i32 2, !dbg !73
  %v50 = load float, ptr %v49, align 4, !dbg !73
  %v51 = fmul contract float %v48, %v50, !dbg !72
  %v52 = icmp ult i64 %v39, 4, !dbg !74
  br i1 %v52, label %bb10, label %bb29, !dbg !74
bb10:
  br label %bb11, !dbg !74
bb11:
  %v53 = getelementptr inbounds float, ptr %__cuda_oxide_local_x3130370931360967726176697479095b6633323b20345d_, i64 %v39, !dbg !74
  %v54 = load float, ptr %v53, align 4, !dbg !74
  %v55 = fadd contract float %v51, %v54, !dbg !75
  %v56 = fmul contract float %v55, %v26, !dbg !75
  %v57 = fadd contract float %v45, %v56, !dbg !76
  %v58 = icmp ult i64 %v39, 4, !dbg !77
  br i1 %v58, label %bb12, label %bb30, !dbg !77
bb12:
  br label %bb13, !dbg !77
bb13:
  %v59 = fmul contract float %v57, %v33, !dbg !77
  %v60 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 1, !dbg !77
  %v61 = getelementptr inbounds float, ptr %v60, i64 %v39, !dbg !77
  store float %v59, ptr %v61, align 4, !dbg !77
  %v62 = icmp ult i64 %v39, 4, !dbg !78
  br i1 %v62, label %bb14, label %bb31, !dbg !78
bb14:
  br label %bb15, !dbg !78
bb15:
  %v63 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 1, !dbg !78
  %v64 = getelementptr inbounds float, ptr %v63, i64 %v39, !dbg !78
  %v65 = load float, ptr %v64, align 4, !dbg !78
  %v66 = fmul contract float %v65, %v26, !dbg !78
  %v67 = icmp ult i64 %v39, 4, !dbg !79
  br i1 %v67, label %bb16, label %bb32, !dbg !79
bb16:
  br label %bb17, !dbg !79
bb17:
  %v68 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 0, !dbg !79
  %v69 = getelementptr inbounds float, ptr %v68, i64 %v39, !dbg !79
  %v70 = load float, ptr %v69, align 4, !dbg !79
  %v71 = fadd contract float %v70, %v66, !dbg !79
  %v72 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 0, !dbg !79
  %v73 = getelementptr inbounds float, ptr %v72, i64 %v39, !dbg !79
  store float %v71, ptr %v73, align 4, !dbg !79
  %v74 = icmp ult i64 %v39, 4, !dbg !80
  br i1 %v74, label %bb18, label %bb33, !dbg !80
bb18:
  br label %bb19, !dbg !80
bb19:
  %v75 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 3, !dbg !80
  %v76 = getelementptr inbounds float, ptr %v75, i64 %v39, !dbg !80
  %v77 = load float, ptr %v76, align 4, !dbg !80
  %v78 = icmp ult i64 %v39, 4, !dbg !81
  br i1 %v78, label %bb20, label %bb34, !dbg !81
bb20:
  br label %bb21, !dbg !81
bb21:
  %v79 = getelementptr inbounds float, ptr %__cuda_oxide_local_x31303509313609616374696f6e095b6633323b20345d_, i64 %v39, !dbg !81
  %v80 = load float, ptr %v79, align 4, !dbg !81
  %v81 = getelementptr inbounds { [4 x float], float, float, float, float, float, [3 x float] }, ptr %__cuda_oxide_local_x3409343809706172616d730953746570506172616d73_, i32 0, i32 3, !dbg !82
  %v82 = load float, ptr %v81, align 8, !dbg !82
  %v83 = fmul contract float %v80, %v82, !dbg !81
  %v84 = fmul contract float %v83, %v26, !dbg !81
  %v85 = fadd contract float %v77, %v84, !dbg !83
  %v86 = icmp ult i64 %v39, 4, !dbg !84
  br i1 %v86, label %bb22, label %bb35, !dbg !84
bb22:
  br label %bb23, !dbg !84
bb23:
  %v87 = fmul contract float %v85, %v38, !dbg !84
  %v88 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 3, !dbg !84
  %v89 = getelementptr inbounds float, ptr %v88, i64 %v39, !dbg !84
  store float %v87, ptr %v89, align 4, !dbg !84
  %v90 = add i64 %v39, 1, !dbg !85
  br label %bb4, !dbg !69
bb24:
  %v91 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 3, !dbg !86
  %v92 = getelementptr inbounds float, ptr %v91, i64 0, !dbg !86
  %v93 = load float, ptr %v92, align 16, !dbg !86
  %v94 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 3, !dbg !88
  %v95 = getelementptr inbounds float, ptr %v94, i64 1, !dbg !88
  %v96 = load float, ptr %v95, align 4, !dbg !88
  %v97 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 3, !dbg !90
  %v98 = getelementptr inbounds float, ptr %v97, i64 2, !dbg !90
  %v99 = load float, ptr %v98, align 8, !dbg !90
  %v100 = fmul contract float 0.5, %v26, !dbg !92
  %v101 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !94
  %v102 = getelementptr inbounds float, ptr %v101, i64 0, !dbg !94
  %v103 = load float, ptr %v102, align 16, !dbg !94
  %v104 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !95
  %v105 = getelementptr inbounds float, ptr %v104, i64 1, !dbg !95
  %v106 = load float, ptr %v105, align 4, !dbg !95
  %v107 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !96
  %v108 = getelementptr inbounds float, ptr %v107, i64 2, !dbg !96
  %v109 = load float, ptr %v108, align 8, !dbg !96
  %v110 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !97
  %v111 = getelementptr inbounds float, ptr %v110, i64 3, !dbg !97
  %v112 = load float, ptr %v111, align 4, !dbg !97
  %v113 = fmul contract float %v93, %v112, !dbg !99
  %v114 = fmul contract float %v96, %v109, !dbg !100
  %v115 = fadd contract float %v113, %v114, !dbg !99
  %v116 = fmul contract float %v99, %v106, !dbg !101
  %v117 = fsub contract float %v115, %v116, !dbg !102
  %v118 = fmul contract float %v100, %v117, !dbg !103
  %v119 = fadd contract float %v103, %v118, !dbg !104
  %v120 = fneg float %v93, !dbg !106
  %v121 = fmul contract float %v120, %v109, !dbg !106
  %v122 = fmul contract float %v96, %v112, !dbg !107
  %v123 = fadd contract float %v121, %v122, !dbg !106
  %v124 = fmul contract float %v99, %v103, !dbg !108
  %v125 = fadd contract float %v123, %v124, !dbg !109
  %v126 = fmul contract float %v100, %v125, !dbg !110
  %v127 = fadd contract float %v106, %v126, !dbg !111
  %v128 = fmul contract float %v93, %v106, !dbg !113
  %v129 = fmul contract float %v96, %v103, !dbg !114
  %v130 = fsub contract float %v128, %v129, !dbg !113
  %v131 = fmul contract float %v99, %v112, !dbg !115
  %v132 = fadd contract float %v130, %v131, !dbg !116
  %v133 = fmul contract float %v100, %v132, !dbg !117
  %v134 = fadd contract float %v109, %v133, !dbg !118
  %v135 = fmul contract float %v93, %v103, !dbg !120
  %v136 = fmul contract float %v96, %v106, !dbg !121
  %v137 = fadd contract float %v135, %v136, !dbg !120
  %v138 = fmul contract float %v99, %v109, !dbg !122
  %v139 = fadd contract float %v137, %v138, !dbg !123
  %v140 = fmul contract float %v100, %v139, !dbg !124
  %v141 = fsub contract float %v112, %v140, !dbg !125
  %v142 = fmul contract float %v119, %v119, !dbg !127
  %v143 = fmul contract float %v127, %v127, !dbg !128
  %v144 = fadd contract float %v142, %v143, !dbg !127
  %v145 = fmul contract float %v134, %v134, !dbg !129
  %v146 = fadd contract float %v144, %v145, !dbg !127
  %v147 = fmul contract float %v141, %v141, !dbg !130
  %v148 = fadd contract float %v146, %v147, !dbg !127
  %v149 = call float asm "rsqrt.approx.f32 $0, $1;", "=f,f"(float %v148), !dbg !131
  br label %bb25, !dbg !131
bb25:
  %v150 = fmul contract float %v119, %v149, !dbg !133
  %v151 = fmul contract float %v127, %v149, !dbg !134
  %v152 = fmul contract float %v134, %v149, !dbg !135
  %v153 = fmul contract float %v141, %v149, !dbg !136
  %v154 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !137
  %v155 = getelementptr inbounds float, ptr %v154, i64 0, !dbg !137
  store float %v150, ptr %v155, align 16, !dbg !137
  %v156 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !137
  %v157 = getelementptr inbounds float, ptr %v156, i64 1, !dbg !137
  store float %v151, ptr %v157, align 4, !dbg !137
  %v158 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !137
  %v159 = getelementptr inbounds float, ptr %v158, i64 2, !dbg !137
  store float %v152, ptr %v159, align 8, !dbg !137
  %v160 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 2, !dbg !137
  %v161 = getelementptr inbounds float, ptr %v160, i64 3, !dbg !137
  store float %v153, ptr %v161, align 4, !dbg !137
  %v162 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 0, !dbg !138
  %v163 = getelementptr inbounds float, ptr %v162, i64 3, !dbg !138
  store float 0.0, ptr %v163, align 4, !dbg !138
  %v164 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 1, !dbg !139
  %v165 = getelementptr inbounds float, ptr %v164, i64 3, !dbg !139
  store float 0.0, ptr %v165, align 4, !dbg !139
  %v166 = getelementptr inbounds { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, i32 0, i32 3, !dbg !140
  %v167 = getelementptr inbounds float, ptr %v166, i64 3, !dbg !140
  store float 0.0, ptr %v167, align 4, !dbg !140
  %v168 = load { [4 x float], [4 x float], [4 x float], [4 x float] }, ptr %__cuda_oxide_local_x313109363409737461746509426f64795374617465_, align 16, !dbg !141
  store { [4 x float], [4 x float], [4 x float], [4 x float] } %v168, ptr %v18, align 16, !dbg !141
  br label %bb26, !dbg !45
bb26:
  ret void, !dbg !45
bb27:
  call void @llvm.trap() #0, !dbg !71
  unreachable, !dbg !71
bb28:
  call void @llvm.trap() #0, !dbg !72
  unreachable, !dbg !72
bb29:
  call void @llvm.trap() #0, !dbg !74
  unreachable, !dbg !74
bb30:
  call void @llvm.trap() #0, !dbg !77
  unreachable, !dbg !77
bb31:
  call void @llvm.trap() #0, !dbg !78
  unreachable, !dbg !78
bb32:
  call void @llvm.trap() #0, !dbg !79
  unreachable, !dbg !79
bb33:
  call void @llvm.trap() #0, !dbg !80
  unreachable, !dbg !80
bb34:
  call void @llvm.trap() #0, !dbg !81
  unreachable, !dbg !81
bb35:
  call void @llvm.trap() #0, !dbg !84
  unreachable, !dbg !84
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare float @llvm.nvvm.fmin.f(float, float)
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #0
declare float @llvm.nvvm.fmax.f(float, float)

define ptx_kernel void @reduce_images(ptr %v0, ptr %v1, i32 %v2) #0 !dbg !142 {
entry:
  %v3 = alloca [4 x float], align 4
  br label %bb0
bb0:
  %v4 = phi ptr [ %v0, %entry ]
  %v5 = phi ptr [ %v1, %entry ]
  %v6 = phi i32 [ %v2, %entry ]
  %v7 = addrspacecast ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_b194c80a4296d25c to ptr, !dbg !143
  br label %bb1, !dbg !143
bb1:
  %v8 = addrspacecast ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_e2553c23b4905f70 to ptr, !dbg !145
  br label %bb2, !dbg !145
bb2:
  %v9 = addrspacecast ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_6b287ec6666962c3 to ptr, !dbg !147
  br label %bb3, !dbg !147
bb3:
  %v10 = addrspacecast ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_880cddad8d2c4a2f to ptr, !dbg !149
  br label %bb4, !dbg !149
bb4:
  %v11 = addrspacecast ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_01b97a1b1cfda569 to ptr, !dbg !151
  br label %bb5, !dbg !151
bb5:
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0, !dbg !153
  br label %bb6, !dbg !153
bb6:
  %v13 = zext i32 %v12 to i64, !dbg !153
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0, !dbg !155
  br label %bb7, !dbg !155
bb7:
  %v15 = zext i32 %v14 to i64, !dbg !155
  %v16 = zext i32 %v6 to i64, !dbg !157
  %v17 = mul i64 %v15, 3, !dbg !159
  %v18 = mul i64 %v17, %v16, !dbg !159
  br label %bb8, !dbg !167
bb8:
  %v19 = phi float [ 0.0, %bb7 ], [ %v44, %bb22 ]
  %v20 = phi float [ 0.0, %bb7 ], [ %v45, %bb22 ]
  %v21 = phi float [ 0.0, %bb7 ], [ %v46, %bb22 ]
  %v22 = phi float [ 0x7FF0000000000000, %bb7 ], [ %v47, %bb22 ]
  %v23 = phi float [ 0xFFF0000000000000, %bb7 ], [ %v107, %bb22 ]
  %v24 = phi i64 [ %v13, %bb7 ], [ %v108, %bb22 ]
  %v25 = icmp ult i64 %v24, %v16, !dbg !168
  %v26 = xor i1 %v25, 1, !dbg !168
  br i1 %v26, label %bb10, label %bb9, !dbg !168
bb9:
  %v27 = add i64 %v18, %v24, !dbg !169
  %v28 = getelementptr inbounds float, ptr %v4, i64 %v27, !dbg !174
  %v29 = load float, ptr %v28, align 4, !dbg !175
  %v30 = add i64 %v18, %v16, !dbg !176
  %v31 = add i64 %v30, %v24, !dbg !176
  %v32 = getelementptr inbounds float, ptr %v4, i64 %v31, !dbg !174
  %v33 = load float, ptr %v32, align 4, !dbg !177
  %v34 = mul i64 2, %v16, !dbg !178
  %v35 = add i64 %v18, %v34, !dbg !179
  %v36 = add i64 %v35, %v24, !dbg !179
  %v37 = getelementptr inbounds float, ptr %v4, i64 %v36, !dbg !174
  %v38 = load float, ptr %v37, align 4, !dbg !180
  %v39 = fmul contract float 0.2125999927520752, %v29, !dbg !182
  %v40 = fmul contract float 0.7152000069618225, %v33, !dbg !183
  %v41 = fadd contract float %v39, %v40, !dbg !182
  %v42 = fmul contract float 0.0722000002861023, %v38, !dbg !184
  %v43 = fadd contract float %v41, %v42, !dbg !182
  %v44 = fadd contract float %v19, %v29, !dbg !186
  %v45 = fadd contract float %v20, %v33, !dbg !187
  %v46 = fadd contract float %v21, %v38, !dbg !188
  %v47 = call float @llvm.nvvm.fmin.f(float %v22, float %v43) #0, !dbg !192
  br label %bb21, !dbg !192
bb10:
  %v48 = getelementptr inbounds float, ptr %v7, i64 %v13, !dbg !200
  store float %v19, ptr %v48, align 4, !dbg !201
  %v49 = getelementptr inbounds float, ptr %v8, i64 %v13, !dbg !200
  store float %v20, ptr %v49, align 4, !dbg !202
  %v50 = getelementptr inbounds float, ptr %v9, i64 %v13, !dbg !200
  store float %v21, ptr %v50, align 4, !dbg !203
  %v51 = getelementptr inbounds float, ptr %v10, i64 %v13, !dbg !200
  store float %v22, ptr %v51, align 4, !dbg !204
  %v52 = getelementptr inbounds float, ptr %v11, i64 %v13, !dbg !200
  store float %v23, ptr %v52, align 4, !dbg !205
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0, !dbg !206
  br label %bb11, !dbg !206
bb11:
  br label %bb12, !dbg !207
bb12:
  %v54 = phi i64 [ 128, %bb11 ], [ %v80, %bb17 ]
  %v55 = icmp ugt i64 %v54, 0, !dbg !208
  %v56 = xor i1 %v55, 1, !dbg !208
  br i1 %v56, label %bb18, label %bb13, !dbg !208
bb13:
  %v57 = icmp ult i64 %v13, %v54, !dbg !209
  %v58 = xor i1 %v57, 1, !dbg !209
  br i1 %v58, label %bb15, label %bb14, !dbg !209
bb14:
  %v59 = add i64 %v13, %v54, !dbg !210
  %v60 = getelementptr inbounds float, ptr %v7, i64 %v59, !dbg !200
  %v61 = load float, ptr %v60, align 4, !dbg !211
  %v62 = load float, ptr %v48, align 4, !dbg !212
  %v63 = fadd contract float %v62, %v61, !dbg !212
  store float %v63, ptr %v48, align 4, !dbg !212
  %v64 = add i64 %v13, %v54, !dbg !213
  %v65 = getelementptr inbounds float, ptr %v8, i64 %v64, !dbg !200
  %v66 = load float, ptr %v65, align 4, !dbg !214
  %v67 = load float, ptr %v49, align 4, !dbg !215
  %v68 = fadd contract float %v67, %v66, !dbg !215
  store float %v68, ptr %v49, align 4, !dbg !215
  %v69 = add i64 %v13, %v54, !dbg !216
  %v70 = getelementptr inbounds float, ptr %v9, i64 %v69, !dbg !200
  %v71 = load float, ptr %v70, align 4, !dbg !217
  %v72 = load float, ptr %v50, align 4, !dbg !218
  %v73 = fadd contract float %v72, %v71, !dbg !218
  store float %v73, ptr %v50, align 4, !dbg !218
  %v74 = load float, ptr %v51, align 4, !dbg !219
  %v75 = add i64 %v13, %v54, !dbg !220
  %v76 = getelementptr inbounds float, ptr %v10, i64 %v75, !dbg !200
  %v77 = load float, ptr %v76, align 4, !dbg !221
  %v78 = call float @llvm.nvvm.fmin.f(float %v74, float %v77) #0, !dbg !192
  br label %bb23, !dbg !192
bb15:
  br label %bb16, !dbg !222
bb16:
  call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #0, !dbg !223
  br label %bb17, !dbg !223
bb17:
  %v80 = udiv i64 %v54, 2, !dbg !224
  br label %bb12, !dbg !207
bb18:
  %v81 = icmp eq i64 %v13, 0, !dbg !225
  br i1 %v81, label %bb19, label %bb20, !dbg !225
bb19:
  %v82 = uitofp i32 %v6 to float, !dbg !226
  %v83 = fdiv contract float 1.0, %v82, !dbg !227
  %v84 = load float, ptr %v7, align 4, !dbg !228
  %v85 = fmul contract float %v84, %v83, !dbg !228
  %v86 = load float, ptr %v8, align 4, !dbg !229
  %v87 = fmul contract float %v86, %v83, !dbg !229
  %v88 = load float, ptr %v9, align 4, !dbg !230
  %v89 = fmul contract float %v88, %v83, !dbg !230
  %v90 = insertvalue [4 x float] undef, float %v85, 0
  %v91 = insertvalue [4 x float] %v90, float %v87, 1
  %v92 = insertvalue [4 x float] %v91, float %v89, 2
  %v93 = insertvalue [4 x float] %v92, float 0.0, 3, !dbg !231
  %v94 = fmul contract float 0.2125999927520752, %v85, !dbg !232
  %v95 = fmul contract float 0.7152000069618225, %v87, !dbg !233
  %v96 = fadd contract float %v94, %v95, !dbg !232
  %v97 = fmul contract float 0.0722000002861023, %v89, !dbg !234
  %v98 = fadd contract float %v96, %v97, !dbg !232
  %v99 = load float, ptr %v10, align 4, !dbg !235
  %v100 = load float, ptr %v11, align 4, !dbg !236
  %v101 = insertvalue { [4 x float], float, float, float, i32 } undef, [4 x float] %v93, 0
  %v102 = insertvalue { [4 x float], float, float, float, i32 } %v101, float %v98, 1
  %v103 = insertvalue { [4 x float], float, float, float, i32 } %v102, float %v99, 2
  %v104 = insertvalue { [4 x float], float, float, float, i32 } %v103, float %v100, 3
  %v105 = insertvalue { [4 x float], float, float, float, i32 } %v104, i32 %v6, 4, !dbg !237
  %v106 = getelementptr inbounds { [4 x float], float, float, float, i32 }, ptr %v5, i64 %v15, !dbg !200
  store { [4 x float], float, float, float, i32 } %v105, ptr %v106, align 16, !dbg !238
  br label %bb20, !dbg !239
bb20:
  ret void, !dbg !240
bb21:
  %v107 = call float @llvm.nvvm.fmax.f(float %v23, float %v43) #0, !dbg !243
  br label %bb22, !dbg !243
bb22:
  %v108 = add i64 %v24, 256, !dbg !244
  br label %bb8, !dbg !167
bb23:
  store float %v78, ptr %v51, align 4, !dbg !245
  %v109 = load float, ptr %v52, align 4, !dbg !246
  %v110 = add i64 %v13, %v54, !dbg !247
  %v111 = getelementptr inbounds float, ptr %v11, i64 %v110, !dbg !200
  %v112 = load float, ptr %v111, align 4, !dbg !248
  %v113 = call float @llvm.nvvm.fmax.f(float %v109, float %v112) #0, !dbg !243
  br label %bb24, !dbg !243
bb24:
  store float %v113, ptr %v52, align 4, !dbg !249
  br label %bb16, !dbg !222
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define i64 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics(ptr %v0) alwaysinline #0 !dbg !251 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0, !dbg !252
  br label %bb1, !dbg !252
bb1:
  %v3 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0, !dbg !253
  br label %bb2, !dbg !253
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0, !dbg !254
  br label %bb7, !dbg !254
bb3:
  %v5 = xor i1 %v20, 1, !dbg !256
  br i1 %v5, label %bb5, label %bb4, !dbg !256
bb4:
  %v6 = icmp ne i64 %v19, 18446744073709551615, !dbg !257
  br label %bb6, !dbg !256
bb5:
  br label %bb6, !dbg !256
bb6:
  %v7 = phi i1 [ %v6, %bb4 ], [ 0, %bb5 ]
  %v8 = xor i1 %v7, 1, !dbg !261
  br i1 %v8, label %bb14, label %bb13, !dbg !261
bb7:
  %v9 = zext i32 %v2 to i64, !dbg !264
  %v10 = zext i32 %v3 to i64, !dbg !265
  %v11 = zext i32 %v4 to i64, !dbg !266
  %v12 = icmp eq i64 %v10, 0, !dbg !267
  br i1 %v12, label %bb10, label %bb8, !dbg !267
bb8:
  %v13 = sub i64 18446744073709551615, %v11, !dbg !268
  %v14 = udiv i64 %v13, %v10, !dbg !268
  %v15 = icmp ugt i64 %v9, %v14, !dbg !269
  %v16 = xor i1 %v15, 1, !dbg !269
  br i1 %v16, label %bb11, label %bb9, !dbg !269
bb9:
  br label %bb10, !dbg !270
bb10:
  br label %bb12, !dbg !271
bb11:
  %v17 = mul i64 %v9, %v10, !dbg !272
  %v18 = add i64 %v17, %v11, !dbg !272
  br label %bb12, !dbg !271
bb12:
  %v19 = phi i64 [ 18446744073709551615, %bb10 ], [ %v18, %bb11 ]
  %v20 = call i1 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics(ptr %v1) #0, !dbg !256
  br label %bb3, !dbg !256
bb13:
  %v21 = icmp eq i64 %v19, 18446744073709551615, !dbg !273
  br i1 %v21, label %bb14, label %bb15, !dbg !273
bb14:
  br label %bb15, !dbg !274
bb15:
  %v22 = phi i64 [ %v19, %bb13 ], [ 18446744073709551615, %bb14 ]
  ret i64 %v22, !dbg !275
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.y()
declare i32 @llvm.nvvm.read.ptx.sreg.ntid.z()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.z()

define i1 @_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics(ptr %v0) alwaysinline #0 !dbg !276 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = icmp eq i8 0, 1, !dbg !277
  %v3 = xor i1 %v2, 1, !dbg !277
  br i1 %v3, label %bb2, label %bb1, !dbg !277
bb1:
  br label %bb8, !dbg !278
bb2:
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0, !dbg !279
  br label %bb3, !dbg !279
bb3:
  %v5 = icmp eq i32 %v4, 1, !dbg !279
  br i1 %v5, label %bb4, label %bb5, !dbg !279
bb4:
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #0, !dbg !280
  br label %bb6, !dbg !280
bb5:
  br label %bb7, !dbg !279
bb6:
  %v7 = icmp eq i32 %v6, 1, !dbg !280
  br label %bb7, !dbg !279
bb7:
  %v8 = phi i1 [ 0, %bb5 ], [ %v7, %bb6 ]
  br label %bb8, !dbg !278
bb8:
  %v9 = phi i1 [ 1, %bb1 ], [ %v8, %bb7 ]
  %v10 = xor i1 %v2, 1, !dbg !282
  br i1 %v10, label %bb9, label %bb10, !dbg !282
bb9:
  %v11 = icmp eq i8 0, 2, !dbg !283
  %v12 = xor i1 %v11, 1, !dbg !283
  br i1 %v12, label %bb11, label %bb10, !dbg !283
bb10:
  br label %bb17, !dbg !284
bb11:
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #0, !dbg !285
  br label %bb12, !dbg !285
bb12:
  %v14 = icmp eq i32 %v13, 1, !dbg !285
  br i1 %v14, label %bb13, label %bb14, !dbg !285
bb13:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #0, !dbg !286
  br label %bb15, !dbg !286
bb14:
  br label %bb16, !dbg !285
bb15:
  %v16 = icmp eq i32 %v15, 1, !dbg !286
  br label %bb16, !dbg !285
bb16:
  %v17 = phi i1 [ 0, %bb14 ], [ %v16, %bb15 ]
  br label %bb17, !dbg !284
bb17:
  %v18 = phi i1 [ 1, %bb10 ], [ %v17, %bb16 ]
  %v19 = xor i1 %v9, 1, !dbg !288
  br i1 %v19, label %bb19, label %bb18, !dbg !288
bb18:
  br label %bb20, !dbg !288
bb19:
  br label %bb20, !dbg !288
bb20:
  %v20 = phi i1 [ %v18, %bb18 ], [ 0, %bb19 ]
  ret i1 %v20, !dbg !289
}


@llvm.used = appending global [3 x ptr] [ptr @integrate, ptr @reduce_images, ptr @rgba8_to_nchw], section "llvm.metadata"

attributes #0 = { convergent }

!290 = !{ptr @integrate, !"align", i32 262160}
!nvvm.annotations = !{!290}

!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!291, !292}
!291 = !{i32 2, !"Dwarf Version", i32 2}
!292 = !{i32 2, !"Debug Info Version", i32 3}
!0 = !DIFile(filename: "kernels.rs", directory: "/home/jaren/visualencoder/sim/crates/sim-physics/src")
!1 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !0, producer: "cuda-oxide", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DISubroutineType(types: !{null})
!3 = distinct !DISubprogram(name: "rgba8_to_nchw", linkageName: "rgba8_to_nchw", scope: !0, file: !0, line: 76, type: !2, scopeLine: 76, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!4 = !DILexicalBlock(scope: !3, file: !0, line: 81, column: 7)
!5 = !DILocation(line: 82, column: 17, scope: !4)
!6 = !DILexicalBlock(scope: !4, file: !0, line: 82, column: 9)
!7 = !DILocation(line: 83, column: 17, scope: !6)
!8 = !DILocation(line: 83, column: 12, scope: !6)
!9 = !DILocation(line: 98, column: 6, scope: !3)
!10 = !DILocation(line: 86, column: 21, scope: !6)
!11 = !DILexicalBlock(scope: !6, file: !0, line: 86, column: 9)
!12 = !DILocation(line: 87, column: 22, scope: !11)
!13 = !DILexicalBlock(scope: !11, file: !0, line: 87, column: 9)
!14 = !DILocation(line: 88, column: 21, scope: !13)
!15 = !DILexicalBlock(scope: !13, file: !0, line: 88, column: 9)
!16 = !DILocation(line: 89, column: 22, scope: !15)
!17 = !DILexicalBlock(scope: !15, file: !0, line: 89, column: 9)
!18 = !DILexicalBlock(scope: !17, file: !0, line: 90, column: 9)
!19 = !DILocation(line: 94, column: 42, scope: !18)
!20 = !DIFile(filename: "const_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!21 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const u8>::add", scope: !20, file: !20, line: 837, type: !2, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!22 = !DILocation(line: 96, column: 50, scope: !18)
!23 = !DILocation(line: 871, column: 18, scope: !21, inlinedAt: !22)
!24 = !DILocation(line: 94, column: 32, scope: !18)
!25 = !DIFile(filename: "mut_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!26 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut f32>::add", scope: !25, file: !25, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!27 = !DILocation(line: 96, column: 18, scope: !18)
!28 = !DILocation(line: 970, column: 18, scope: !26, inlinedAt: !27)
!29 = !DILocation(line: 94, column: 13, scope: !18)
!30 = !DILocation(line: 95, column: 50, scope: !18)
!31 = !DILocation(line: 95, column: 40, scope: !18)
!32 = !DILocation(line: 95, column: 22, scope: !18)
!33 = !DILocation(line: 95, column: 13, scope: !18)
!34 = !DILocation(line: 96, column: 54, scope: !18)
!35 = !DILocation(line: 96, column: 44, scope: !18)
!36 = !DILocation(line: 96, column: 31, scope: !18)
!37 = !DILocation(line: 96, column: 22, scope: !18)
!38 = !DILocation(line: 96, column: 13, scope: !18)
!39 = distinct !DISubprogram(name: "integrate", linkageName: "integrate", scope: !0, file: !0, line: 21, type: !2, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!40 = !DILexicalBlock(scope: !39, file: !0, line: 26, column: 7)
!41 = !DILocation(line: 27, column: 17, scope: !40)
!42 = !DILexicalBlock(scope: !40, file: !0, line: 27, column: 9)
!43 = !DILocation(line: 28, column: 17, scope: !42)
!44 = !DILocation(line: 28, column: 12, scope: !42)
!45 = !DILocation(line: 67, column: 6, scope: !39)
!46 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut sim_state::BodyState>::add", scope: !25, file: !25, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!47 = !DILocation(line: 32, column: 42, scope: !42)
!48 = !DILocation(line: 970, column: 18, scope: !46, inlinedAt: !47)
!49 = !DILocation(line: 32, column: 34, scope: !42)
!50 = !DILexicalBlock(scope: !42, file: !0, line: 32, column: 9)
!51 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const sim_state::BodyAction>::add", scope: !20, file: !20, line: 837, type: !2, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!52 = !DILocation(line: 33, column: 40, scope: !50)
!53 = !DILocation(line: 871, column: 18, scope: !51, inlinedAt: !52)
!54 = !DILocation(line: 33, column: 31, scope: !50)
!55 = !DILexicalBlock(scope: !50, file: !0, line: 33, column: 9)
!56 = !DILocation(line: 34, column: 18, scope: !55)
!57 = !DILexicalBlock(scope: !55, file: !0, line: 34, column: 9)
!58 = !DILocation(line: 35, column: 23, scope: !57)
!59 = !DILexicalBlock(scope: !57, file: !0, line: 35, column: 9)
!60 = !DILocation(line: 36, column: 41, scope: !59)
!61 = !DILocation(line: 36, column: 34, scope: !59)
!62 = !DILocation(line: 36, column: 28, scope: !59)
!63 = !DILexicalBlock(scope: !59, file: !0, line: 36, column: 9)
!64 = !DILocation(line: 37, column: 42, scope: !63)
!65 = !DILocation(line: 37, column: 35, scope: !63)
!66 = !DILocation(line: 37, column: 29, scope: !63)
!67 = !DILexicalBlock(scope: !63, file: !0, line: 37, column: 9)
!68 = !DILexicalBlock(scope: !67, file: !0, line: 38, column: 9)
!69 = !DILocation(line: 39, column: 9, scope: !68)
!70 = !DILocation(line: 39, column: 15, scope: !68)
!71 = !DILocation(line: 40, column: 37, scope: !68)
!72 = !DILocation(line: 41, column: 20, scope: !68)
!73 = !DILocation(line: 41, column: 41, scope: !68)
!74 = !DILocation(line: 41, column: 65, scope: !68)
!75 = !DILocation(line: 41, column: 19, scope: !68)
!76 = !DILocation(line: 40, column: 36, scope: !68)
!77 = !DILocation(line: 40, column: 13, scope: !68)
!78 = !DILocation(line: 43, column: 37, scope: !68)
!79 = !DILocation(line: 43, column: 13, scope: !68)
!80 = !DILocation(line: 44, column: 45, scope: !68)
!81 = !DILocation(line: 45, column: 19, scope: !68)
!82 = !DILocation(line: 45, column: 41, scope: !68)
!83 = !DILocation(line: 44, column: 44, scope: !68)
!84 = !DILocation(line: 44, column: 13, scope: !68)
!85 = !DILocation(line: 47, column: 13, scope: !68)
!86 = !DILocation(line: 51, column: 18, scope: !68)
!87 = !DILexicalBlock(scope: !68, file: !0, line: 51, column: 9)
!88 = !DILocation(line: 52, column: 18, scope: !87)
!89 = !DILexicalBlock(scope: !87, file: !0, line: 52, column: 9)
!90 = !DILocation(line: 53, column: 18, scope: !89)
!91 = !DILexicalBlock(scope: !89, file: !0, line: 53, column: 9)
!92 = !DILocation(line: 54, column: 23, scope: !91)
!93 = !DILexicalBlock(scope: !91, file: !0, line: 54, column: 9)
!94 = !DILocation(line: 55, column: 14, scope: !93)
!95 = !DILocation(line: 55, column: 18, scope: !93)
!96 = !DILocation(line: 55, column: 22, scope: !93)
!97 = !DILocation(line: 55, column: 26, scope: !93)
!98 = !DILexicalBlock(scope: !93, file: !0, line: 55, column: 9)
!99 = !DILocation(line: 56, column: 33, scope: !98)
!100 = !DILocation(line: 56, column: 43, scope: !98)
!101 = !DILocation(line: 56, column: 53, scope: !98)
!102 = !DILocation(line: 56, column: 32, scope: !98)
!103 = !DILocation(line: 56, column: 22, scope: !98)
!104 = !DILocation(line: 56, column: 17, scope: !98)
!105 = !DILexicalBlock(scope: !98, file: !0, line: 56, column: 9)
!106 = !DILocation(line: 57, column: 33, scope: !105)
!107 = !DILocation(line: 57, column: 44, scope: !105)
!108 = !DILocation(line: 57, column: 54, scope: !105)
!109 = !DILocation(line: 57, column: 32, scope: !105)
!110 = !DILocation(line: 57, column: 22, scope: !105)
!111 = !DILocation(line: 57, column: 17, scope: !105)
!112 = !DILexicalBlock(scope: !105, file: !0, line: 57, column: 9)
!113 = !DILocation(line: 58, column: 33, scope: !112)
!114 = !DILocation(line: 58, column: 43, scope: !112)
!115 = !DILocation(line: 58, column: 53, scope: !112)
!116 = !DILocation(line: 58, column: 32, scope: !112)
!117 = !DILocation(line: 58, column: 22, scope: !112)
!118 = !DILocation(line: 58, column: 17, scope: !112)
!119 = !DILexicalBlock(scope: !112, file: !0, line: 58, column: 9)
!120 = !DILocation(line: 59, column: 33, scope: !119)
!121 = !DILocation(line: 59, column: 43, scope: !119)
!122 = !DILocation(line: 59, column: 53, scope: !119)
!123 = !DILocation(line: 59, column: 32, scope: !119)
!124 = !DILocation(line: 59, column: 22, scope: !119)
!125 = !DILocation(line: 59, column: 17, scope: !119)
!126 = !DILexicalBlock(scope: !119, file: !0, line: 59, column: 9)
!127 = !DILocation(line: 60, column: 52, scope: !126)
!128 = !DILocation(line: 60, column: 60, scope: !126)
!129 = !DILocation(line: 60, column: 68, scope: !126)
!130 = !DILocation(line: 60, column: 76, scope: !126)
!131 = !DILocation(line: 60, column: 28, scope: !126)
!132 = !DILexicalBlock(scope: !126, file: !0, line: 60, column: 9)
!133 = !DILocation(line: 61, column: 27, scope: !132)
!134 = !DILocation(line: 61, column: 45, scope: !132)
!135 = !DILocation(line: 61, column: 63, scope: !132)
!136 = !DILocation(line: 61, column: 81, scope: !132)
!137 = !DILocation(line: 61, column: 9, scope: !132)
!138 = !DILocation(line: 62, column: 9, scope: !132)
!139 = !DILocation(line: 63, column: 9, scope: !132)
!140 = !DILocation(line: 64, column: 9, scope: !132)
!141 = !DILocation(line: 66, column: 18, scope: !132)
!142 = distinct !DISubprogram(name: "reduce_images", linkageName: "reduce_images", scope: !0, file: !0, line: 108, type: !2, scopeLine: 108, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!143 = !DILocation(line: 121, column: 35, scope: !142)
!144 = !DILexicalBlock(scope: !142, file: !0, line: 121, column: 9)
!145 = !DILocation(line: 122, column: 37, scope: !144)
!146 = !DILexicalBlock(scope: !144, file: !0, line: 122, column: 9)
!147 = !DILocation(line: 123, column: 36, scope: !146)
!148 = !DILexicalBlock(scope: !146, file: !0, line: 123, column: 9)
!149 = !DILocation(line: 124, column: 39, scope: !148)
!150 = !DILexicalBlock(scope: !148, file: !0, line: 124, column: 9)
!151 = !DILocation(line: 125, column: 39, scope: !150)
!152 = !DILexicalBlock(scope: !150, file: !0, line: 125, column: 9)
!153 = !DILocation(line: 127, column: 20, scope: !152)
!154 = !DILexicalBlock(scope: !152, file: !0, line: 127, column: 9)
!155 = !DILocation(line: 128, column: 22, scope: !154)
!156 = !DILexicalBlock(scope: !154, file: !0, line: 128, column: 9)
!157 = !DILocation(line: 129, column: 21, scope: !156)
!158 = !DILexicalBlock(scope: !156, file: !0, line: 129, column: 9)
!159 = !DILocation(line: 130, column: 20, scope: !158)
!160 = !DILexicalBlock(scope: !158, file: !0, line: 130, column: 9)
!161 = !DILexicalBlock(scope: !160, file: !0, line: 131, column: 9)
!162 = !DILexicalBlock(scope: !161, file: !0, line: 132, column: 9)
!163 = !DILexicalBlock(scope: !162, file: !0, line: 133, column: 9)
!164 = !DILexicalBlock(scope: !163, file: !0, line: 134, column: 9)
!165 = !DILexicalBlock(scope: !164, file: !0, line: 135, column: 9)
!166 = !DILexicalBlock(scope: !165, file: !0, line: 136, column: 9)
!167 = !DILocation(line: 137, column: 9, scope: !166)
!168 = !DILocation(line: 137, column: 15, scope: !166)
!169 = !DILocation(line: 139, column: 39, scope: !166)
!170 = !DILexicalBlock(scope: !166, file: !0, line: 139, column: 13)
!171 = !DILexicalBlock(scope: !170, file: !0, line: 140, column: 13)
!172 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const f32>::add", scope: !20, file: !20, line: 837, type: !2, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!173 = !DILocation(line: 141, column: 35, scope: !171)
!174 = !DILocation(line: 871, column: 18, scope: !172, inlinedAt: !173)
!175 = !DILocation(line: 139, column: 30, scope: !166)
!176 = !DILocation(line: 140, column: 39, scope: !170)
!177 = !DILocation(line: 140, column: 30, scope: !170)
!178 = !DILocation(line: 141, column: 46, scope: !171)
!179 = !DILocation(line: 141, column: 39, scope: !171)
!180 = !DILocation(line: 141, column: 30, scope: !171)
!181 = !DILexicalBlock(scope: !171, file: !0, line: 141, column: 13)
!182 = !DILocation(line: 142, column: 29, scope: !181)
!183 = !DILocation(line: 142, column: 42, scope: !181)
!184 = !DILocation(line: 142, column: 55, scope: !181)
!185 = !DILexicalBlock(scope: !181, file: !0, line: 142, column: 13)
!186 = !DILocation(line: 143, column: 13, scope: !185)
!187 = !DILocation(line: 144, column: 13, scope: !185)
!188 = !DILocation(line: 145, column: 13, scope: !185)
!189 = !DIFile(filename: "f32.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/num")
!190 = distinct !DISubprogram(name: "core::f32::<impl f32>::min", scope: !189, file: !189, line: 1117, type: !2, scopeLine: 1117, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!191 = !DILocation(line: 146, column: 31, scope: !185)
!192 = !DILocation(line: 1118, column: 9, scope: !190, inlinedAt: !191)
!193 = !DILexicalBlock(scope: !166, file: !0, line: 159, column: 9)
!194 = !DILexicalBlock(scope: !193, file: !0, line: 178, column: 13)
!195 = !DILexicalBlock(scope: !194, file: !0, line: 181, column: 17)
!196 = !DILexicalBlock(scope: !195, file: !0, line: 182, column: 17)
!197 = !DILexicalBlock(scope: !196, file: !0, line: 183, column: 17)
!198 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut image::ImageStatistics>::add", scope: !25, file: !25, line: 936, type: !2, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!199 = !DILocation(line: 184, column: 29, scope: !197)
!200 = !DILocation(line: 970, column: 18, scope: !198, inlinedAt: !199)
!201 = !DILocation(line: 152, column: 13, scope: !166)
!202 = !DILocation(line: 153, column: 13, scope: !166)
!203 = !DILocation(line: 154, column: 13, scope: !166)
!204 = !DILocation(line: 155, column: 13, scope: !166)
!205 = !DILocation(line: 156, column: 13, scope: !166)
!206 = !DILocation(line: 158, column: 9, scope: !166)
!207 = !DILocation(line: 160, column: 9, scope: !193)
!208 = !DILocation(line: 160, column: 15, scope: !193)
!209 = !DILocation(line: 161, column: 16, scope: !193)
!210 = !DILocation(line: 165, column: 62, scope: !193)
!211 = !DILocation(line: 165, column: 46, scope: !193)
!212 = !DILocation(line: 165, column: 21, scope: !193)
!213 = !DILocation(line: 166, column: 66, scope: !193)
!214 = !DILocation(line: 166, column: 48, scope: !193)
!215 = !DILocation(line: 166, column: 21, scope: !193)
!216 = !DILocation(line: 167, column: 64, scope: !193)
!217 = !DILocation(line: 167, column: 47, scope: !193)
!218 = !DILocation(line: 167, column: 21, scope: !193)
!219 = !DILocation(line: 169, column: 25, scope: !193)
!220 = !DILocation(line: 169, column: 77, scope: !193)
!221 = !DILocation(line: 169, column: 57, scope: !193)
!222 = !DILocation(line: 161, column: 13, scope: !193)
!223 = !DILocation(line: 174, column: 13, scope: !193)
!224 = !DILocation(line: 175, column: 13, scope: !193)
!225 = !DILocation(line: 177, column: 12, scope: !193)
!226 = !DILocation(line: 178, column: 42, scope: !193)
!227 = !DILocation(line: 178, column: 36, scope: !193)
!228 = !DILocation(line: 181, column: 25, scope: !194)
!229 = !DILocation(line: 182, column: 25, scope: !195)
!230 = !DILocation(line: 183, column: 25, scope: !196)
!231 = !DILocation(line: 185, column: 31, scope: !197)
!232 = !DILocation(line: 186, column: 37, scope: !197)
!233 = !DILocation(line: 186, column: 50, scope: !197)
!234 = !DILocation(line: 186, column: 63, scope: !197)
!235 = !DILocation(line: 187, column: 36, scope: !197)
!236 = !DILocation(line: 188, column: 36, scope: !197)
!237 = !DILocation(line: 184, column: 43, scope: !197)
!238 = !DILocation(line: 184, column: 17, scope: !197)
!239 = !DILocation(line: 177, column: 9, scope: !193)
!240 = !DILocation(line: 193, column: 6, scope: !142)
!241 = distinct !DISubprogram(name: "core::f32::<impl f32>::max", scope: !189, file: !189, line: 1090, type: !2, scopeLine: 1090, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!242 = !DILocation(line: 147, column: 31, scope: !185)
!243 = !DILocation(line: 1091, column: 9, scope: !241, inlinedAt: !242)
!244 = !DILocation(line: 148, column: 13, scope: !185)
!245 = !DILocation(line: 168, column: 21, scope: !193)
!246 = !DILocation(line: 171, column: 25, scope: !193)
!247 = !DILocation(line: 171, column: 77, scope: !193)
!248 = !DILocation(line: 171, column: 57, scope: !193)
!249 = !DILocation(line: 170, column: 21, scope: !193)
!250 = !DIFile(filename: "thread.rs", directory: "/home/jaren/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/348895e/cuda-oxide/crates/cuda-device/src")
!251 = distinct !DISubprogram(name: "cuda_device::__internal::index_1d::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics", scope: !250, file: !250, line: 648, type: !2, scopeLine: 648, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!252 = !DILocation(line: 652, column: 13, scope: !251)
!253 = !DILocation(line: 653, column: 13, scope: !251)
!254 = !DILocation(line: 654, column: 13, scope: !251)
!255 = !DILexicalBlock(scope: !251, file: !250, line: 651, column: 9)
!256 = !DILocation(line: 656, column: 21, scope: !255)
!257 = !DILocation(line: 656, column: 54, scope: !255)
!258 = !DILexicalBlock(scope: !255, file: !250, line: 656, column: 9)
!259 = distinct !DISubprogram(name: "cuda_device::ThreadIndex::<'_>::new::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", scope: !250, file: !250, line: 303, type: !2, scopeLine: 303, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!260 = !DILocation(line: 657, column: 18, scope: !258)
!261 = !DILocation(line: 312, column: 21, scope: !259, inlinedAt: !260)
!262 = distinct !DISubprogram(name: "cuda_device::__internal::checked_axis", scope: !250, file: !250, line: 601, type: !2, scopeLine: 601, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!263 = !DILocation(line: 651, column: 19, scope: !251)
!264 = !DILocation(line: 602, column: 21, scope: !262, inlinedAt: !263)
!265 = !DILocation(line: 603, column: 26, scope: !262, inlinedAt: !263)
!266 = !DILocation(line: 604, column: 22, scope: !262, inlinedAt: !263)
!267 = !DILocation(line: 605, column: 12, scope: !262, inlinedAt: !263)
!268 = !DILocation(line: 605, column: 39, scope: !262, inlinedAt: !263)
!269 = !DILocation(line: 605, column: 31, scope: !262, inlinedAt: !263)
!270 = !DILocation(line: 605, column: 74, scope: !262, inlinedAt: !263)
!271 = !DILocation(line: 605, column: 9, scope: !262, inlinedAt: !263)
!272 = !DILocation(line: 608, column: 13, scope: !262, inlinedAt: !263)
!273 = !DILocation(line: 312, column: 30, scope: !259, inlinedAt: !260)
!274 = !DILocation(line: 312, column: 18, scope: !259, inlinedAt: !260)
!275 = !DILocation(line: 658, column: 6, scope: !251)
!276 = distinct !DISubprogram(name: "cuda_device::__internal::one_dimensional_launch::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics", scope: !250, file: !250, line: 613, type: !2, scopeLine: 613, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !{})
!277 = !DILocation(line: 616, column: 28, scope: !276)
!278 = !DILocation(line: 616, column: 25, scope: !276)
!279 = !DILocation(line: 619, column: 13, scope: !276)
!280 = !DILocation(line: 619, column: 41, scope: !276)
!281 = !DILexicalBlock(scope: !276, file: !250, line: 616, column: 9)
!282 = !DILocation(line: 621, column: 28, scope: !281)
!283 = !DILocation(line: 621, column: 59, scope: !281)
!284 = !DILocation(line: 621, column: 25, scope: !281)
!285 = !DILocation(line: 624, column: 13, scope: !281)
!286 = !DILocation(line: 624, column: 41, scope: !281)
!287 = !DILexicalBlock(scope: !281, file: !250, line: 621, column: 9)
!288 = !DILocation(line: 626, column: 9, scope: !287)
!289 = !DILocation(line: 627, column: 6, scope: !276)
