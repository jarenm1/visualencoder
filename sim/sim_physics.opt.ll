; ModuleID = '/home/jaren/visualencoder/sim/sim_physics.ll'
source_filename = "sim_physics"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@__shared_mem_f6621971da3d7ca3_01b97a1b1cfda569 = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_f6621971da3d7ca3_880cddad8d2c4a2f = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_f6621971da3d7ca3_6b287ec6666962c3 = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_f6621971da3d7ca3_e2553c23b4905f70 = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_f6621971da3d7ca3_b194c80a4296d25c = local_unnamed_addr addrspace(3) global [256 x float] zeroinitializer, align 4
@llvm.used = appending global [3 x ptr] [ptr @integrate, ptr @reduce_images, ptr @rgba8_to_nchw], section "llvm.metadata"

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #0

; Function Attrs: convergent nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @rgba8_to_nchw(ptr nofree readonly captures(none) %v0, ptr nofree writeonly captures(none) %v1, i32 %v2, i32 %v3) #1 !dbg !4 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7, !dbg !8
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7, !dbg !13
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7, !dbg !14
  %v9.i = zext nneg i32 %v2.i to i64, !dbg !15
  %v10.i = zext nneg i32 %v3.i to i64, !dbg !18
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i, !dbg !19
  %v11.i = zext nneg i32 %v4.i to i64, !dbg !20
  %v18.i = add nuw nsw i64 %v17.i, %v11.i, !dbg !19
  %v4.i1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #7, !dbg !21
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #7, !dbg !21
  %v5.i = icmp eq i32 %v4.i1, 1, !dbg !21
  %v7.i = icmp eq i32 %v6.i, 1, !dbg !21
  %v8.not.not.i = and i1 %v5.i, %v7.i, !dbg !25
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #7, !dbg !25
  %v14.i = icmp eq i32 %v13.i, 1, !dbg !25
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #7, !dbg !25
  %v16.i = icmp eq i32 %v15.i, 1, !dbg !25
  %v17.i2 = and i1 %v14.i, %v16.i, !dbg !25
  %.v18.i = and i1 %v8.not.not.i, %v17.i2, !dbg !27
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1, !dbg !29
  %v10 = zext i32 %v2 to i64, !dbg !33
  %v11.not = icmp ult i64 %v22.i, %v10, !dbg !35
  br i1 %v11.not, label %bb3, label %bb8, !dbg !35

bb3:                                              ; preds = %entry
  %v14.not = icmp eq i32 %v3, 0, !dbg !36
  br i1 %v14.not, label %bb9, label %bb5, !dbg !36

bb5:                                              ; preds = %bb3
  %v13 = zext i32 %v3 to i64, !dbg !38
  %v16.lhs.trunc = trunc nuw i64 %v22.i to i32, !dbg !36
  %v3.frozen = freeze i32 %v3, !dbg !36
  %v163 = udiv i32 %v16.lhs.trunc, %v3.frozen, !dbg !36
  %v16.zext = zext i32 %v163 to i64, !dbg !36
  %0 = mul i32 %v163, %v3.frozen, !dbg !39
  %v184.decomposed = sub i32 %v16.lhs.trunc, %0, !dbg !39
  %v18.zext = zext i32 %v184.decomposed to i64, !dbg !39
  %v19 = mul nuw nsw i64 %v13, 3, !dbg !41
  %v20 = mul i64 %v19, %v16.zext, !dbg !41
  %v22 = shl nuw nsw i64 %v22.i, 2, !dbg !43
  %v23 = getelementptr inbounds nuw i8, ptr %v0, i64 %v22, !dbg !46
  %v24 = load i8, ptr %v23, align 1, !dbg !50
  %v25 = uitofp i8 %v24 to float, !dbg !50
  %1 = getelementptr [4 x i8], ptr %v1, i64 %v20, !dbg !51
  %v26 = getelementptr [4 x i8], ptr %1, i64 %v18.zext, !dbg !51
  %v27 = fmul nnan contract float %v25, f0x3B808081, !dbg !55
  store float %v27, ptr %v26, align 4, !dbg !55
  %v29 = getelementptr i8, ptr %v23, i64 1, !dbg !46
  %v30 = load i8, ptr %v29, align 1, !dbg !56
  %v31 = uitofp i8 %v30 to float, !dbg !56
  %v33 = getelementptr [4 x i8], ptr %v26, i64 %v13, !dbg !51
  %v34 = fmul nnan contract float %v31, f0x3B808081, !dbg !57
  store float %v34, ptr %v33, align 4, !dbg !57
  %v36 = getelementptr i8, ptr %v23, i64 2, !dbg !46
  %v37 = load i8, ptr %v36, align 1, !dbg !58
  %v38 = uitofp i8 %v37 to float, !dbg !58
  %v41.idx = shl nuw nsw i64 %v13, 3, !dbg !51
  %v41 = getelementptr i8, ptr %v26, i64 %v41.idx, !dbg !51
  %v42 = fmul nnan contract float %v38, f0x3B808081, !dbg !59
  store float %v42, ptr %v41, align 4, !dbg !59
  br label %bb8, !dbg !60

bb8:                                              ; preds = %entry, %bb5
  ret void, !dbg !60

bb9:                                              ; preds = %bb3
  tail call void @llvm.trap() #7, !dbg !36
  unreachable, !dbg !36
}

; Function Attrs: convergent nounwind memory(argmem: readwrite)
define ptx_kernel void @integrate(ptr nofree captures(none) %v0, ptr nofree readonly captures(none) %v1, i32 %v2, { [4 x float], float, float, float, float, float, [3 x float] } alignstack(16) %v3) #2 !dbg !61 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7, !dbg !62
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7, !dbg !65
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7, !dbg !66
  %v9.i = zext nneg i32 %v2.i to i64, !dbg !67
  %v10.i = zext nneg i32 %v3.i to i64, !dbg !69
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i, !dbg !70
  %v11.i = zext nneg i32 %v4.i to i64, !dbg !71
  %v18.i = add nuw nsw i64 %v17.i, %v11.i, !dbg !70
  %v4.i84 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #7, !dbg !72
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #7, !dbg !72
  %v5.i = icmp eq i32 %v4.i84, 1, !dbg !72
  %v7.i = icmp eq i32 %v6.i, 1, !dbg !72
  %v8.not.not.i = and i1 %v5.i, %v7.i, !dbg !74
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #7, !dbg !74
  %v14.i = icmp eq i32 %v13.i, 1, !dbg !74
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #7, !dbg !74
  %v16.i = icmp eq i32 %v15.i, 1, !dbg !74
  %v17.i85 = and i1 %v14.i, %v16.i, !dbg !74
  %.v18.i = and i1 %v8.not.not.i, %v17.i85, !dbg !75
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1, !dbg !76
  %v15 = zext i32 %v2 to i64, !dbg !78
  %v16.not = icmp ult i64 %v22.i, %v15, !dbg !80
  br i1 %v16.not, label %bb3, label %bb26, !dbg !80

bb3:                                              ; preds = %entry
  %v3.fca.3.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 3
  %v3.fca.2.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 2
  %v3.fca.1.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 1
  %v3.fca.5.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 5
  %v3.fca.4.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 4
  %v3.fca.0.2.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 0, 2
  %v3.fca.0.1.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 0, 1
  %v3.fca.0.0.extract = extractvalue { [4 x float], float, float, float, float, float, [3 x float] } %v3, 0, 0
  %v18 = getelementptr inbounds nuw [64 x i8], ptr %v0, i64 %v22.i, !dbg !81
  %v19.unpack.unpack = load float, ptr %v18, align 16, !dbg !84
  %v19.unpack.elt12 = getelementptr inbounds nuw i8, ptr %v18, i64 4, !dbg !84
  %v19.unpack.unpack13 = load float, ptr %v19.unpack.elt12, align 4, !dbg !84
  %v19.unpack.elt14 = getelementptr inbounds nuw i8, ptr %v18, i64 8, !dbg !84
  %v19.unpack.unpack15 = load float, ptr %v19.unpack.elt14, align 8, !dbg !84
  %v19.unpack.elt16 = getelementptr inbounds nuw i8, ptr %v18, i64 12, !dbg !84
  %v19.elt5 = getelementptr inbounds nuw i8, ptr %v18, i64 16, !dbg !84
  %v19.unpack6.unpack = load float, ptr %v19.elt5, align 16, !dbg !84
  %v19.unpack6.elt19 = getelementptr inbounds nuw i8, ptr %v18, i64 20, !dbg !84
  %v19.unpack6.unpack20 = load float, ptr %v19.unpack6.elt19, align 4, !dbg !84
  %v19.unpack6.elt21 = getelementptr inbounds nuw i8, ptr %v18, i64 24, !dbg !84
  %v19.unpack6.unpack22 = load float, ptr %v19.unpack6.elt21, align 8, !dbg !84
  %v19.unpack6.elt23 = getelementptr inbounds nuw i8, ptr %v18, i64 28, !dbg !84
  %v19.elt7 = getelementptr inbounds nuw i8, ptr %v18, i64 32, !dbg !84
  %v19.unpack8.unpack = load float, ptr %v19.elt7, align 16, !dbg !84
  %v19.unpack8.elt26 = getelementptr inbounds nuw i8, ptr %v18, i64 36, !dbg !84
  %v19.unpack8.unpack27 = load float, ptr %v19.unpack8.elt26, align 4, !dbg !84
  %v19.unpack8.elt28 = getelementptr inbounds nuw i8, ptr %v18, i64 40, !dbg !84
  %v19.unpack8.unpack29 = load float, ptr %v19.unpack8.elt28, align 8, !dbg !84
  %v19.unpack8.elt30 = getelementptr inbounds nuw i8, ptr %v18, i64 44, !dbg !84
  %v19.unpack8.unpack31 = load float, ptr %v19.unpack8.elt30, align 4, !dbg !84
  %v19.elt9 = getelementptr inbounds nuw i8, ptr %v18, i64 48, !dbg !84
  %v19.unpack10.unpack = load float, ptr %v19.elt9, align 16, !dbg !84
  %v19.unpack10.elt33 = getelementptr inbounds nuw i8, ptr %v18, i64 52, !dbg !84
  %v19.unpack10.unpack34 = load float, ptr %v19.unpack10.elt33, align 4, !dbg !84
  %v19.unpack10.elt35 = getelementptr inbounds nuw i8, ptr %v18, i64 56, !dbg !84
  %v19.unpack10.unpack36 = load float, ptr %v19.unpack10.elt35, align 8, !dbg !84
  %v19.unpack10.elt37 = getelementptr inbounds nuw i8, ptr %v18, i64 60, !dbg !84
  %v20 = getelementptr inbounds nuw [32 x i8], ptr %v1, i64 %v22.i, !dbg !85
  %v22.unpack = load float, ptr %v20, align 16, !dbg !89
  %v22.elt40 = getelementptr inbounds nuw i8, ptr %v20, i64 4, !dbg !89
  %v22.unpack41 = load float, ptr %v22.elt40, align 4, !dbg !89
  %v22.elt42 = getelementptr inbounds nuw i8, ptr %v20, i64 8, !dbg !89
  %v22.unpack43 = load float, ptr %v22.elt42, align 8, !dbg !89
  %v23 = getelementptr inbounds nuw i8, ptr %v20, i64 16, !dbg !89
  %v24.unpack = load float, ptr %v23, align 16, !dbg !89
  %v24.elt47 = getelementptr inbounds nuw i8, ptr %v20, i64 20, !dbg !89
  %v24.unpack48 = load float, ptr %v24.elt47, align 4, !dbg !89
  %v24.elt49 = getelementptr inbounds nuw i8, ptr %v20, i64 24, !dbg !89
  %v24.unpack50 = load float, ptr %v24.elt49, align 8, !dbg !89
  %v31 = fmul contract float %v3.fca.4.extract, %v3.fca.1.extract, !dbg !90
  %v32 = fadd contract float %v31, 1.000000e+00, !dbg !94
  %v33 = fdiv contract float 1.000000e+00, %v32, !dbg !95
  %v36 = fmul contract float %v3.fca.5.extract, %v3.fca.1.extract, !dbg !96
  %v37 = fadd contract float %v36, 1.000000e+00, !dbg !98
  %v38 = fdiv contract float 1.000000e+00, %v37, !dbg !99
  %v51 = fmul contract float %v3.fca.2.extract, %v22.unpack, !dbg !100
  %v55 = fadd contract float %v51, %v3.fca.0.0.extract, !dbg !103
  %v56 = fmul contract float %v3.fca.1.extract, %v55, !dbg !103
  %v57 = fadd contract float %v19.unpack6.unpack, %v56, !dbg !104
  %v59 = fmul contract float %v33, %v57, !dbg !105
  %v66 = fmul contract float %v3.fca.1.extract, %v59, !dbg !106
  %v71 = fadd contract float %v19.unpack.unpack, %v66, !dbg !107
  %v83 = fmul contract float %v3.fca.3.extract, %v24.unpack, !dbg !108
  %v84 = fmul contract float %v3.fca.1.extract, %v83, !dbg !108
  %v85 = fadd contract float %v19.unpack10.unpack, %v84, !dbg !109
  %v87 = fmul contract float %v38, %v85, !dbg !110
  %v51.1 = fmul contract float %v3.fca.2.extract, %v22.unpack41, !dbg !100
  %v55.1 = fadd contract float %v51.1, %v3.fca.0.1.extract, !dbg !103
  %v56.1 = fmul contract float %v3.fca.1.extract, %v55.1, !dbg !103
  %v57.1 = fadd contract float %v19.unpack6.unpack20, %v56.1, !dbg !104
  %v59.1 = fmul contract float %v33, %v57.1, !dbg !105
  %v66.1 = fmul contract float %v3.fca.1.extract, %v59.1, !dbg !106
  %v71.1 = fadd contract float %v19.unpack.unpack13, %v66.1, !dbg !107
  %v83.1 = fmul contract float %v3.fca.3.extract, %v24.unpack48, !dbg !108
  %v84.1 = fmul contract float %v3.fca.1.extract, %v83.1, !dbg !108
  %v85.1 = fadd contract float %v19.unpack10.unpack34, %v84.1, !dbg !109
  %v87.1 = fmul contract float %v38, %v85.1, !dbg !110
  %v51.2 = fmul contract float %v3.fca.2.extract, %v22.unpack43, !dbg !100
  %v55.2 = fadd contract float %v51.2, %v3.fca.0.2.extract, !dbg !103
  %v56.2 = fmul contract float %v3.fca.1.extract, %v55.2, !dbg !103
  %v57.2 = fadd contract float %v19.unpack6.unpack22, %v56.2, !dbg !104
  %v59.2 = fmul contract float %v33, %v57.2, !dbg !105
  %v66.2 = fmul contract float %v3.fca.1.extract, %v59.2, !dbg !106
  %v71.2 = fadd contract float %v19.unpack.unpack15, %v66.2, !dbg !107
  %v83.2 = fmul contract float %v3.fca.3.extract, %v24.unpack50, !dbg !108
  %v84.2 = fmul contract float %v3.fca.1.extract, %v83.2, !dbg !108
  %v85.2 = fadd contract float %v19.unpack10.unpack36, %v84.2, !dbg !109
  %v87.2 = fmul contract float %v38, %v85.2, !dbg !110
  %v100 = fmul contract float %v3.fca.1.extract, 5.000000e-01, !dbg !111
  %v113 = fmul contract float %v87, %v19.unpack8.unpack31, !dbg !115
  %v114 = fmul contract float %v87.1, %v19.unpack8.unpack29, !dbg !118
  %v115 = fadd contract float %v114, %v113, !dbg !115
  %v116 = fmul contract float %v87.2, %v19.unpack8.unpack27, !dbg !119
  %v117 = fsub contract float %v115, %v116, !dbg !120
  %v118 = fmul contract float %v100, %v117, !dbg !121
  %v119 = fadd contract float %v19.unpack8.unpack, %v118, !dbg !122
  %v122 = fmul contract float %v87.1, %v19.unpack8.unpack31, !dbg !123
  %0 = fmul contract float %v87, %v19.unpack8.unpack29, !dbg !125
  %v123 = fsub contract float %v122, %0, !dbg !125
  %v124 = fmul contract float %v87.2, %v19.unpack8.unpack, !dbg !126
  %v125 = fadd contract float %v124, %v123, !dbg !127
  %v126 = fmul contract float %v100, %v125, !dbg !128
  %v127 = fadd contract float %v19.unpack8.unpack27, %v126, !dbg !129
  %v128 = fmul contract float %v87, %v19.unpack8.unpack27, !dbg !130
  %v129 = fmul contract float %v87.1, %v19.unpack8.unpack, !dbg !132
  %v130 = fsub contract float %v128, %v129, !dbg !130
  %v131 = fmul contract float %v87.2, %v19.unpack8.unpack31, !dbg !133
  %v132 = fadd contract float %v130, %v131, !dbg !134
  %v133 = fmul contract float %v100, %v132, !dbg !135
  %v134 = fadd contract float %v19.unpack8.unpack29, %v133, !dbg !136
  %v135 = fmul contract float %v87, %v19.unpack8.unpack, !dbg !137
  %v136 = fmul contract float %v87.1, %v19.unpack8.unpack27, !dbg !139
  %v137 = fadd contract float %v135, %v136, !dbg !137
  %v138 = fmul contract float %v87.2, %v19.unpack8.unpack29, !dbg !140
  %v139 = fadd contract float %v137, %v138, !dbg !141
  %v140 = fmul contract float %v100, %v139, !dbg !142
  %v141 = fsub contract float %v19.unpack8.unpack31, %v140, !dbg !143
  %v142 = fmul contract float %v119, %v119, !dbg !144
  %v143 = fmul contract float %v127, %v127, !dbg !146
  %v144 = fadd contract float %v142, %v143, !dbg !144
  %v145 = fmul contract float %v134, %v134, !dbg !147
  %v146 = fadd contract float %v145, %v144, !dbg !144
  %v147 = fmul contract float %v141, %v141, !dbg !148
  %v148 = fadd contract float %v147, %v146, !dbg !144
  %v149 = tail call float asm "rsqrt.approx.f32 $0, $1;", "=f,f"(float %v148) #8, !dbg !149
  %v150 = fmul contract float %v149, %v119, !dbg !150
  %v151 = fmul contract float %v149, %v127, !dbg !152
  %v152 = fmul contract float %v149, %v134, !dbg !153
  %v153 = fmul contract float %v149, %v141, !dbg !154
  store float %v71, ptr %v18, align 16, !dbg !155
  store float %v71.1, ptr %v19.unpack.elt12, align 4, !dbg !155
  store float %v71.2, ptr %v19.unpack.elt14, align 8, !dbg !155
  store float 0.000000e+00, ptr %v19.unpack.elt16, align 4, !dbg !155
  store float %v59, ptr %v19.elt5, align 16, !dbg !155
  store float %v59.1, ptr %v19.unpack6.elt19, align 4, !dbg !155
  store float %v59.2, ptr %v19.unpack6.elt21, align 8, !dbg !155
  store float 0.000000e+00, ptr %v19.unpack6.elt23, align 4, !dbg !155
  store float %v150, ptr %v19.elt7, align 16, !dbg !155
  store float %v151, ptr %v19.unpack8.elt26, align 4, !dbg !155
  store float %v152, ptr %v19.unpack8.elt28, align 8, !dbg !155
  store float %v153, ptr %v19.unpack8.elt30, align 4, !dbg !155
  store float %v87, ptr %v19.elt9, align 16, !dbg !155
  store float %v87.1, ptr %v19.unpack10.elt33, align 4, !dbg !155
  store float %v87.2, ptr %v19.unpack10.elt35, align 8, !dbg !155
  store float 0.000000e+00, ptr %v19.unpack10.elt37, align 4, !dbg !155
  br label %bb26, !dbg !156

bb26:                                             ; preds = %entry, %bb3
  ret void, !dbg !156
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #3

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.cta.sync.aligned.all(i32) #4

; Function Attrs: convergent norecurse nounwind
define ptx_kernel void @reduce_images(ptr nofree readonly captures(none) %v0, ptr nofree writeonly captures(none) %v1, i32 %v2) #5 !dbg !157 {
entry:
  %v12 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7, !dbg !158
  %v13 = zext nneg i32 %v12 to i64, !dbg !158
  %v14 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7, !dbg !164
  %v15 = zext nneg i32 %v14 to i64, !dbg !164
  %v16 = zext i32 %v2 to i64, !dbg !166
  %v25.not17 = icmp ult i32 %v12, %v2, !dbg !168
  br i1 %v25.not17, label %bb9.lr.ph, label %bb10, !dbg !168

bb9.lr.ph:                                        ; preds = %entry
  %v17 = mul nuw nsw i64 %v15, 3, !dbg !177
  %v18 = mul i64 %v17, %v16, !dbg !177
  %0 = getelementptr [4 x i8], ptr %v0, i64 %v18
  %1 = getelementptr [4 x i8], ptr %0, i64 %v16
  %v181 = add nuw nsw i64 %v17, 2
  %v35 = mul i64 %v181, %v16
  %2 = getelementptr [4 x i8], ptr %v0, i64 %v35
  br label %bb9, !dbg !168

bb9:                                              ; preds = %bb9.lr.ph, %bb9
  %v2423 = phi i64 [ %v13, %bb9.lr.ph ], [ %v108, %bb9 ]
  %v2322 = phi float [ -inf, %bb9.lr.ph ], [ %v107, %bb9 ]
  %v2221 = phi float [ +inf, %bb9.lr.ph ], [ %v47, %bb9 ]
  %v2120 = phi float [ 0.000000e+00, %bb9.lr.ph ], [ %v46, %bb9 ]
  %v2019 = phi float [ 0.000000e+00, %bb9.lr.ph ], [ %v45, %bb9 ]
  %v1918 = phi float [ 0.000000e+00, %bb9.lr.ph ], [ %v44, %bb9 ]
  %v28 = getelementptr [4 x i8], ptr %0, i64 %v2423, !dbg !178
  %v29 = load float, ptr %v28, align 4, !dbg !183
  %v32 = getelementptr [4 x i8], ptr %1, i64 %v2423, !dbg !178
  %v33 = load float, ptr %v32, align 4, !dbg !184
  %v37 = getelementptr [4 x i8], ptr %2, i64 %v2423, !dbg !178
  %v38 = load float, ptr %v37, align 4, !dbg !185
  %v39 = fmul contract float %v29, 2.126000e-01, !dbg !186
  %v40 = fmul contract float %v33, 7.152000e-01, !dbg !188
  %v41 = fadd contract float %v39, %v40, !dbg !186
  %v42 = fmul contract float %v38, 7.220000e-02, !dbg !189
  %v43 = fadd contract float %v41, %v42, !dbg !186
  %v44 = fadd contract float %v1918, %v29, !dbg !190
  %v45 = fadd contract float %v2019, %v33, !dbg !192
  %v46 = fadd contract float %v2120, %v38, !dbg !193
  %v47 = tail call float @llvm.minimumnum.f32(float %v2221, float %v43), !dbg !194
  %v107 = tail call float @llvm.maximumnum.f32(float %v2322, float %v43), !dbg !198
  %v108 = add nuw nsw i64 %v2423, 256, !dbg !201
  %v25.not = icmp samesign ult i64 %v108, %v16, !dbg !168
  br i1 %v25.not, label %bb9, label %bb10, !dbg !168

bb10:                                             ; preds = %bb9, %entry
  %v19.lcssa = phi float [ 0.000000e+00, %entry ], [ %v44, %bb9 ]
  %v20.lcssa = phi float [ 0.000000e+00, %entry ], [ %v45, %bb9 ]
  %v21.lcssa = phi float [ 0.000000e+00, %entry ], [ %v46, %bb9 ]
  %v22.lcssa = phi float [ +inf, %entry ], [ %v47, %bb9 ]
  %v23.lcssa = phi float [ -inf, %entry ], [ %v107, %bb9 ]
  %v48 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_b194c80a4296d25c to ptr), i64 %v13, !dbg !202
  store float %v19.lcssa, ptr %v48, align 4, !dbg !210
  %v49 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_e2553c23b4905f70 to ptr), i64 %v13, !dbg !202
  store float %v20.lcssa, ptr %v49, align 4, !dbg !211
  %v50 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_6b287ec6666962c3 to ptr), i64 %v13, !dbg !202
  store float %v21.lcssa, ptr %v50, align 4, !dbg !212
  %v51 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_880cddad8d2c4a2f to ptr), i64 %v13, !dbg !202
  store float %v22.lcssa, ptr %v51, align 4, !dbg !213
  %v52 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_01b97a1b1cfda569 to ptr), i64 %v13, !dbg !202
  store float %v23.lcssa, ptr %v52, align 4, !dbg !214
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #7, !dbg !215
  br label %bb13, !dbg !216

bb13:                                             ; preds = %bb10, %bb16
  %v5428 = phi i64 [ 128, %bb10 ], [ %v802, %bb16 ]
  %v57.not = icmp samesign ugt i64 %v5428, %v13, !dbg !217
  br i1 %v57.not, label %bb14, label %bb16, !dbg !217

bb14:                                             ; preds = %bb13
  %v59 = add nuw nsw i64 %v5428, %v13, !dbg !218
  %v60 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_b194c80a4296d25c to ptr), i64 %v59, !dbg !202
  %v61 = load float, ptr %v60, align 4, !dbg !219
  %v62 = load float, ptr %v48, align 4, !dbg !220
  %v63 = fadd contract float %v61, %v62, !dbg !220
  store float %v63, ptr %v48, align 4, !dbg !220
  %v65 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_e2553c23b4905f70 to ptr), i64 %v59, !dbg !202
  %v66 = load float, ptr %v65, align 4, !dbg !221
  %v67 = load float, ptr %v49, align 4, !dbg !222
  %v68 = fadd contract float %v66, %v67, !dbg !222
  store float %v68, ptr %v49, align 4, !dbg !222
  %v70 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_6b287ec6666962c3 to ptr), i64 %v59, !dbg !202
  %v71 = load float, ptr %v70, align 4, !dbg !223
  %v72 = load float, ptr %v50, align 4, !dbg !224
  %v73 = fadd contract float %v71, %v72, !dbg !224
  store float %v73, ptr %v50, align 4, !dbg !224
  %v74 = load float, ptr %v51, align 4, !dbg !225
  %v76 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_880cddad8d2c4a2f to ptr), i64 %v59, !dbg !202
  %v77 = load float, ptr %v76, align 4, !dbg !226
  %v78 = tail call float @llvm.minimumnum.f32(float %v74, float %v77), !dbg !194
  store float %v78, ptr %v51, align 4, !dbg !227
  %v109 = load float, ptr %v52, align 4, !dbg !228
  %v111 = getelementptr inbounds nuw [4 x i8], ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_01b97a1b1cfda569 to ptr), i64 %v59, !dbg !202
  %v112 = load float, ptr %v111, align 4, !dbg !229
  %v113 = tail call float @llvm.maximumnum.f32(float %v109, float %v112), !dbg !198
  store float %v113, ptr %v52, align 4, !dbg !230
  br label %bb16, !dbg !231

bb16:                                             ; preds = %bb13, %bb14
  tail call void @llvm.nvvm.barrier.cta.sync.aligned.all(i32 0) #7, !dbg !232
  %v802 = lshr i64 %v5428, 1, !dbg !233
  %v55.not = icmp eq i64 %v802, 0, !dbg !216
  br i1 %v55.not, label %bb18, label %bb13, !dbg !216

bb18:                                             ; preds = %bb16
  %v81 = icmp eq i32 %v12, 0, !dbg !234
  br i1 %v81, label %bb19, label %bb20, !dbg !234

bb19:                                             ; preds = %bb18
  %v82 = uitofp i32 %v2 to float, !dbg !235
  %v83 = fdiv contract float 1.000000e+00, %v82, !dbg !236
  %v84 = load float, ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_b194c80a4296d25c to ptr), align 4, !dbg !237
  %v85 = fmul contract float %v83, %v84, !dbg !237
  %v86 = load float, ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_e2553c23b4905f70 to ptr), align 4, !dbg !238
  %v87 = fmul contract float %v83, %v86, !dbg !238
  %v88 = load float, ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_6b287ec6666962c3 to ptr), align 4, !dbg !239
  %v89 = fmul contract float %v83, %v88, !dbg !239
  %v94 = fmul contract float %v85, 2.126000e-01, !dbg !240
  %v95 = fmul contract float %v87, 7.152000e-01, !dbg !241
  %v96 = fadd contract float %v94, %v95, !dbg !240
  %v97 = fmul contract float %v89, 7.220000e-02, !dbg !242
  %v98 = fadd contract float %v96, %v97, !dbg !240
  %v99 = load float, ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_880cddad8d2c4a2f to ptr), align 4, !dbg !243
  %v100 = load float, ptr addrspacecast (ptr addrspace(3) @__shared_mem_f6621971da3d7ca3_01b97a1b1cfda569 to ptr), align 4, !dbg !244
  %v106 = getelementptr inbounds nuw [32 x i8], ptr %v1, i64 %v15, !dbg !202
  store float %v85, ptr %v106, align 16, !dbg !245
  %v106.repack11 = getelementptr inbounds nuw i8, ptr %v106, i64 4, !dbg !245
  store float %v87, ptr %v106.repack11, align 4, !dbg !245
  %v106.repack13 = getelementptr inbounds nuw i8, ptr %v106, i64 8, !dbg !245
  store float %v89, ptr %v106.repack13, align 8, !dbg !245
  %v106.repack15 = getelementptr inbounds nuw i8, ptr %v106, i64 12, !dbg !245
  store float 0.000000e+00, ptr %v106.repack15, align 4, !dbg !245
  %v106.repack3 = getelementptr inbounds nuw i8, ptr %v106, i64 16, !dbg !245
  store float %v98, ptr %v106.repack3, align 16, !dbg !245
  %v106.repack5 = getelementptr inbounds nuw i8, ptr %v106, i64 20, !dbg !245
  store float %v99, ptr %v106.repack5, align 4, !dbg !245
  %v106.repack7 = getelementptr inbounds nuw i8, ptr %v106, i64 24, !dbg !245
  store float %v100, ptr %v106.repack7, align 8, !dbg !245
  %v106.repack9 = getelementptr inbounds nuw i8, ptr %v106, i64 28, !dbg !245
  store i32 %v2, ptr %v106.repack9, align 4, !dbg !245
  br label %bb20, !dbg !246

bb20:                                             ; preds = %bb19, %bb18
  ret void, !dbg !247
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #3

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #3

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.minimumnum.f32(float, float) #6

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maximumnum.f32(float, float) #6

attributes #0 = { cold noreturn nounwind memory(inaccessiblemem: write) "target-cpu"="sm_120" }
attributes #1 = { convergent nounwind memory(argmem: readwrite, inaccessiblemem: write) "target-cpu"="sm_120" }
attributes #2 = { convergent nounwind memory(argmem: readwrite) "target-cpu"="sm_120" }
attributes #3 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="sm_120" }
attributes #4 = { convergent nocallback nounwind "target-cpu"="sm_120" }
attributes #5 = { convergent norecurse nounwind "target-cpu"="sm_120" }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent }
attributes #8 = { nounwind }

!nvvm.annotations = !{}
!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "cuda-oxide", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "kernels.rs", directory: "/home/jaren/visualencoder/sim/crates/sim-physics/src")
!2 = !{i32 2, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "rgba8_to_nchw", linkageName: "rgba8_to_nchw", scope: !1, file: !1, line: 76, type: !5, scopeLine: 76, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{}
!8 = !DILocation(line: 652, column: 13, scope: !9, inlinedAt: !11)
!9 = distinct !DISubprogram(name: "cuda_device::__internal::index_1d::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics", scope: !10, file: !10, line: 648, type: !5, scopeLine: 648, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!10 = !DIFile(filename: "thread.rs", directory: "/home/jaren/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/348895e/cuda-oxide/crates/cuda-device/src")
!11 = distinct !DILocation(line: 82, column: 17, scope: !12)
!12 = !DILexicalBlock(scope: !4, file: !1, line: 81, column: 7)
!13 = !DILocation(line: 653, column: 13, scope: !9, inlinedAt: !11)
!14 = !DILocation(line: 654, column: 13, scope: !9, inlinedAt: !11)
!15 = !DILocation(line: 602, column: 21, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "cuda_device::__internal::checked_axis", scope: !10, file: !10, line: 601, type: !5, scopeLine: 601, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!17 = distinct !DILocation(line: 651, column: 19, scope: !9, inlinedAt: !11)
!18 = !DILocation(line: 603, column: 26, scope: !16, inlinedAt: !17)
!19 = !DILocation(line: 608, column: 13, scope: !16, inlinedAt: !17)
!20 = !DILocation(line: 604, column: 22, scope: !16, inlinedAt: !17)
!21 = !DILocation(line: 619, column: 13, scope: !22, inlinedAt: !23)
!22 = distinct !DISubprogram(name: "cuda_device::__internal::one_dimensional_launch::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsl9uvetKAkWB_11sim_physics", scope: !10, file: !10, line: 613, type: !5, scopeLine: 613, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!23 = distinct !DILocation(line: 656, column: 21, scope: !24, inlinedAt: !11)
!24 = !DILexicalBlock(scope: !9, file: !10, line: 651, column: 9)
!25 = !DILocation(line: 624, column: 13, scope: !26, inlinedAt: !23)
!26 = !DILexicalBlock(scope: !22, file: !10, line: 616, column: 9)
!27 = !DILocation(line: 626, column: 9, scope: !28, inlinedAt: !23)
!28 = !DILexicalBlock(scope: !26, file: !10, line: 621, column: 9)
!29 = !DILocation(line: 312, column: 21, scope: !30, inlinedAt: !31)
!30 = distinct !DISubprogram(name: "cuda_device::ThreadIndex::<'_>::new::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", scope: !10, file: !10, line: 303, type: !5, scopeLine: 303, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!31 = distinct !DILocation(line: 657, column: 18, scope: !32, inlinedAt: !11)
!32 = !DILexicalBlock(scope: !24, file: !10, line: 656, column: 9)
!33 = !DILocation(line: 83, column: 17, scope: !34)
!34 = !DILexicalBlock(scope: !12, file: !1, line: 82, column: 9)
!35 = !DILocation(line: 83, column: 12, scope: !34)
!36 = !DILocation(line: 87, column: 22, scope: !37)
!37 = !DILexicalBlock(scope: !34, file: !1, line: 86, column: 9)
!38 = !DILocation(line: 86, column: 21, scope: !34)
!39 = !DILocation(line: 88, column: 21, scope: !40)
!40 = !DILexicalBlock(scope: !37, file: !1, line: 87, column: 9)
!41 = !DILocation(line: 89, column: 22, scope: !42)
!42 = !DILexicalBlock(scope: !40, file: !1, line: 88, column: 9)
!43 = !DILocation(line: 94, column: 42, scope: !44)
!44 = !DILexicalBlock(scope: !45, file: !1, line: 90, column: 9)
!45 = !DILexicalBlock(scope: !42, file: !1, line: 89, column: 9)
!46 = !DILocation(line: 871, column: 18, scope: !47, inlinedAt: !49)
!47 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const u8>::add", scope: !48, file: !48, line: 837, type: !5, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!48 = !DIFile(filename: "const_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!49 = !DILocation(line: 96, column: 50, scope: !44)
!50 = !DILocation(line: 94, column: 32, scope: !44)
!51 = !DILocation(line: 970, column: 18, scope: !52, inlinedAt: !54)
!52 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut f32>::add", scope: !53, file: !53, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!53 = !DIFile(filename: "mut_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!54 = !DILocation(line: 96, column: 18, scope: !44)
!55 = !DILocation(line: 94, column: 13, scope: !44)
!56 = !DILocation(line: 95, column: 40, scope: !44)
!57 = !DILocation(line: 95, column: 13, scope: !44)
!58 = !DILocation(line: 96, column: 44, scope: !44)
!59 = !DILocation(line: 96, column: 13, scope: !44)
!60 = !DILocation(line: 98, column: 6, scope: !4)
!61 = distinct !DISubprogram(name: "integrate", linkageName: "integrate", scope: !1, file: !1, line: 21, type: !5, scopeLine: 21, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!62 = !DILocation(line: 652, column: 13, scope: !9, inlinedAt: !63)
!63 = distinct !DILocation(line: 27, column: 17, scope: !64)
!64 = !DILexicalBlock(scope: !61, file: !1, line: 26, column: 7)
!65 = !DILocation(line: 653, column: 13, scope: !9, inlinedAt: !63)
!66 = !DILocation(line: 654, column: 13, scope: !9, inlinedAt: !63)
!67 = !DILocation(line: 602, column: 21, scope: !16, inlinedAt: !68)
!68 = distinct !DILocation(line: 651, column: 19, scope: !9, inlinedAt: !63)
!69 = !DILocation(line: 603, column: 26, scope: !16, inlinedAt: !68)
!70 = !DILocation(line: 608, column: 13, scope: !16, inlinedAt: !68)
!71 = !DILocation(line: 604, column: 22, scope: !16, inlinedAt: !68)
!72 = !DILocation(line: 619, column: 13, scope: !22, inlinedAt: !73)
!73 = distinct !DILocation(line: 656, column: 21, scope: !24, inlinedAt: !63)
!74 = !DILocation(line: 624, column: 13, scope: !26, inlinedAt: !73)
!75 = !DILocation(line: 626, column: 9, scope: !28, inlinedAt: !73)
!76 = !DILocation(line: 312, column: 21, scope: !30, inlinedAt: !77)
!77 = distinct !DILocation(line: 657, column: 18, scope: !32, inlinedAt: !63)
!78 = !DILocation(line: 28, column: 17, scope: !79)
!79 = !DILexicalBlock(scope: !64, file: !1, line: 27, column: 9)
!80 = !DILocation(line: 28, column: 12, scope: !79)
!81 = !DILocation(line: 970, column: 18, scope: !82, inlinedAt: !83)
!82 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut sim_state::BodyState>::add", scope: !53, file: !53, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!83 = !DILocation(line: 32, column: 42, scope: !79)
!84 = !DILocation(line: 32, column: 34, scope: !79)
!85 = !DILocation(line: 871, column: 18, scope: !86, inlinedAt: !87)
!86 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const sim_state::BodyAction>::add", scope: !48, file: !48, line: 837, type: !5, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!87 = !DILocation(line: 33, column: 40, scope: !88)
!88 = !DILexicalBlock(scope: !79, file: !1, line: 32, column: 9)
!89 = !DILocation(line: 33, column: 31, scope: !88)
!90 = !DILocation(line: 36, column: 41, scope: !91)
!91 = !DILexicalBlock(scope: !92, file: !1, line: 35, column: 9)
!92 = !DILexicalBlock(scope: !93, file: !1, line: 34, column: 9)
!93 = !DILexicalBlock(scope: !88, file: !1, line: 33, column: 9)
!94 = !DILocation(line: 36, column: 34, scope: !91)
!95 = !DILocation(line: 36, column: 28, scope: !91)
!96 = !DILocation(line: 37, column: 42, scope: !97)
!97 = !DILexicalBlock(scope: !91, file: !1, line: 36, column: 9)
!98 = !DILocation(line: 37, column: 35, scope: !97)
!99 = !DILocation(line: 37, column: 29, scope: !97)
!100 = !DILocation(line: 41, column: 20, scope: !101)
!101 = !DILexicalBlock(scope: !102, file: !1, line: 38, column: 9)
!102 = !DILexicalBlock(scope: !97, file: !1, line: 37, column: 9)
!103 = !DILocation(line: 41, column: 19, scope: !101)
!104 = !DILocation(line: 40, column: 36, scope: !101)
!105 = !DILocation(line: 40, column: 13, scope: !101)
!106 = !DILocation(line: 43, column: 37, scope: !101)
!107 = !DILocation(line: 43, column: 13, scope: !101)
!108 = !DILocation(line: 45, column: 19, scope: !101)
!109 = !DILocation(line: 44, column: 44, scope: !101)
!110 = !DILocation(line: 44, column: 13, scope: !101)
!111 = !DILocation(line: 54, column: 23, scope: !112)
!112 = !DILexicalBlock(scope: !113, file: !1, line: 53, column: 9)
!113 = !DILexicalBlock(scope: !114, file: !1, line: 52, column: 9)
!114 = !DILexicalBlock(scope: !101, file: !1, line: 51, column: 9)
!115 = !DILocation(line: 56, column: 33, scope: !116)
!116 = !DILexicalBlock(scope: !117, file: !1, line: 55, column: 9)
!117 = !DILexicalBlock(scope: !112, file: !1, line: 54, column: 9)
!118 = !DILocation(line: 56, column: 43, scope: !116)
!119 = !DILocation(line: 56, column: 53, scope: !116)
!120 = !DILocation(line: 56, column: 32, scope: !116)
!121 = !DILocation(line: 56, column: 22, scope: !116)
!122 = !DILocation(line: 56, column: 17, scope: !116)
!123 = !DILocation(line: 57, column: 44, scope: !124)
!124 = !DILexicalBlock(scope: !116, file: !1, line: 56, column: 9)
!125 = !DILocation(line: 57, column: 33, scope: !124)
!126 = !DILocation(line: 57, column: 54, scope: !124)
!127 = !DILocation(line: 57, column: 32, scope: !124)
!128 = !DILocation(line: 57, column: 22, scope: !124)
!129 = !DILocation(line: 57, column: 17, scope: !124)
!130 = !DILocation(line: 58, column: 33, scope: !131)
!131 = !DILexicalBlock(scope: !124, file: !1, line: 57, column: 9)
!132 = !DILocation(line: 58, column: 43, scope: !131)
!133 = !DILocation(line: 58, column: 53, scope: !131)
!134 = !DILocation(line: 58, column: 32, scope: !131)
!135 = !DILocation(line: 58, column: 22, scope: !131)
!136 = !DILocation(line: 58, column: 17, scope: !131)
!137 = !DILocation(line: 59, column: 33, scope: !138)
!138 = !DILexicalBlock(scope: !131, file: !1, line: 58, column: 9)
!139 = !DILocation(line: 59, column: 43, scope: !138)
!140 = !DILocation(line: 59, column: 53, scope: !138)
!141 = !DILocation(line: 59, column: 32, scope: !138)
!142 = !DILocation(line: 59, column: 22, scope: !138)
!143 = !DILocation(line: 59, column: 17, scope: !138)
!144 = !DILocation(line: 60, column: 52, scope: !145)
!145 = !DILexicalBlock(scope: !138, file: !1, line: 59, column: 9)
!146 = !DILocation(line: 60, column: 60, scope: !145)
!147 = !DILocation(line: 60, column: 68, scope: !145)
!148 = !DILocation(line: 60, column: 76, scope: !145)
!149 = !DILocation(line: 60, column: 28, scope: !145)
!150 = !DILocation(line: 61, column: 27, scope: !151)
!151 = !DILexicalBlock(scope: !145, file: !1, line: 60, column: 9)
!152 = !DILocation(line: 61, column: 45, scope: !151)
!153 = !DILocation(line: 61, column: 63, scope: !151)
!154 = !DILocation(line: 61, column: 81, scope: !151)
!155 = !DILocation(line: 66, column: 18, scope: !151)
!156 = !DILocation(line: 67, column: 6, scope: !61)
!157 = distinct !DISubprogram(name: "reduce_images", linkageName: "reduce_images", scope: !1, file: !1, line: 108, type: !5, scopeLine: 108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!158 = !DILocation(line: 127, column: 20, scope: !159)
!159 = !DILexicalBlock(scope: !160, file: !1, line: 125, column: 9)
!160 = !DILexicalBlock(scope: !161, file: !1, line: 124, column: 9)
!161 = !DILexicalBlock(scope: !162, file: !1, line: 123, column: 9)
!162 = !DILexicalBlock(scope: !163, file: !1, line: 122, column: 9)
!163 = !DILexicalBlock(scope: !157, file: !1, line: 121, column: 9)
!164 = !DILocation(line: 128, column: 22, scope: !165)
!165 = !DILexicalBlock(scope: !159, file: !1, line: 127, column: 9)
!166 = !DILocation(line: 129, column: 21, scope: !167)
!167 = !DILexicalBlock(scope: !165, file: !1, line: 128, column: 9)
!168 = !DILocation(line: 137, column: 15, scope: !169)
!169 = !DILexicalBlock(scope: !170, file: !1, line: 136, column: 9)
!170 = !DILexicalBlock(scope: !171, file: !1, line: 135, column: 9)
!171 = !DILexicalBlock(scope: !172, file: !1, line: 134, column: 9)
!172 = !DILexicalBlock(scope: !173, file: !1, line: 133, column: 9)
!173 = !DILexicalBlock(scope: !174, file: !1, line: 132, column: 9)
!174 = !DILexicalBlock(scope: !175, file: !1, line: 131, column: 9)
!175 = !DILexicalBlock(scope: !176, file: !1, line: 130, column: 9)
!176 = !DILexicalBlock(scope: !167, file: !1, line: 129, column: 9)
!177 = !DILocation(line: 130, column: 20, scope: !176)
!178 = !DILocation(line: 871, column: 18, scope: !179, inlinedAt: !180)
!179 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const f32>::add", scope: !48, file: !48, line: 837, type: !5, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!180 = !DILocation(line: 141, column: 35, scope: !181)
!181 = !DILexicalBlock(scope: !182, file: !1, line: 140, column: 13)
!182 = !DILexicalBlock(scope: !169, file: !1, line: 139, column: 13)
!183 = !DILocation(line: 139, column: 30, scope: !169)
!184 = !DILocation(line: 140, column: 30, scope: !182)
!185 = !DILocation(line: 141, column: 30, scope: !181)
!186 = !DILocation(line: 142, column: 29, scope: !187)
!187 = !DILexicalBlock(scope: !181, file: !1, line: 141, column: 13)
!188 = !DILocation(line: 142, column: 42, scope: !187)
!189 = !DILocation(line: 142, column: 55, scope: !187)
!190 = !DILocation(line: 143, column: 13, scope: !191)
!191 = !DILexicalBlock(scope: !187, file: !1, line: 142, column: 13)
!192 = !DILocation(line: 144, column: 13, scope: !191)
!193 = !DILocation(line: 145, column: 13, scope: !191)
!194 = !DILocation(line: 1118, column: 9, scope: !195, inlinedAt: !197)
!195 = distinct !DISubprogram(name: "core::f32::<impl f32>::min", scope: !196, file: !196, line: 1117, type: !5, scopeLine: 1117, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!196 = !DIFile(filename: "f32.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/num")
!197 = !DILocation(line: 146, column: 31, scope: !191)
!198 = !DILocation(line: 1091, column: 9, scope: !199, inlinedAt: !200)
!199 = distinct !DISubprogram(name: "core::f32::<impl f32>::max", scope: !196, file: !196, line: 1090, type: !5, scopeLine: 1090, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!200 = !DILocation(line: 147, column: 31, scope: !191)
!201 = !DILocation(line: 148, column: 13, scope: !191)
!202 = !DILocation(line: 970, column: 18, scope: !203, inlinedAt: !204)
!203 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut image::ImageStatistics>::add", scope: !53, file: !53, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!204 = !DILocation(line: 184, column: 29, scope: !205)
!205 = !DILexicalBlock(scope: !206, file: !1, line: 183, column: 17)
!206 = !DILexicalBlock(scope: !207, file: !1, line: 182, column: 17)
!207 = !DILexicalBlock(scope: !208, file: !1, line: 181, column: 17)
!208 = !DILexicalBlock(scope: !209, file: !1, line: 178, column: 13)
!209 = !DILexicalBlock(scope: !169, file: !1, line: 159, column: 9)
!210 = !DILocation(line: 152, column: 13, scope: !169)
!211 = !DILocation(line: 153, column: 13, scope: !169)
!212 = !DILocation(line: 154, column: 13, scope: !169)
!213 = !DILocation(line: 155, column: 13, scope: !169)
!214 = !DILocation(line: 156, column: 13, scope: !169)
!215 = !DILocation(line: 158, column: 9, scope: !169)
!216 = !DILocation(line: 160, column: 15, scope: !209)
!217 = !DILocation(line: 161, column: 16, scope: !209)
!218 = !DILocation(line: 165, column: 62, scope: !209)
!219 = !DILocation(line: 165, column: 46, scope: !209)
!220 = !DILocation(line: 165, column: 21, scope: !209)
!221 = !DILocation(line: 166, column: 48, scope: !209)
!222 = !DILocation(line: 166, column: 21, scope: !209)
!223 = !DILocation(line: 167, column: 47, scope: !209)
!224 = !DILocation(line: 167, column: 21, scope: !209)
!225 = !DILocation(line: 169, column: 25, scope: !209)
!226 = !DILocation(line: 169, column: 57, scope: !209)
!227 = !DILocation(line: 168, column: 21, scope: !209)
!228 = !DILocation(line: 171, column: 25, scope: !209)
!229 = !DILocation(line: 171, column: 57, scope: !209)
!230 = !DILocation(line: 170, column: 21, scope: !209)
!231 = !DILocation(line: 161, column: 13, scope: !209)
!232 = !DILocation(line: 174, column: 13, scope: !209)
!233 = !DILocation(line: 175, column: 13, scope: !209)
!234 = !DILocation(line: 177, column: 12, scope: !209)
!235 = !DILocation(line: 178, column: 42, scope: !209)
!236 = !DILocation(line: 178, column: 36, scope: !209)
!237 = !DILocation(line: 181, column: 25, scope: !208)
!238 = !DILocation(line: 182, column: 25, scope: !207)
!239 = !DILocation(line: 183, column: 25, scope: !206)
!240 = !DILocation(line: 186, column: 37, scope: !205)
!241 = !DILocation(line: 186, column: 50, scope: !205)
!242 = !DILocation(line: 186, column: 63, scope: !205)
!243 = !DILocation(line: 187, column: 36, scope: !205)
!244 = !DILocation(line: 188, column: 36, scope: !205)
!245 = !DILocation(line: 184, column: 17, scope: !205)
!246 = !DILocation(line: 177, column: 9, scope: !209)
!247 = !DILocation(line: 193, column: 6, scope: !157)
