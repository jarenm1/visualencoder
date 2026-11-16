; ModuleID = '/home/jaren/visualencoder/sim/sim_env.ll'
source_filename = "sim_env"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@llvm.used = appending global [3 x ptr] [ptr @control, ptr @finish, ptr @reset], section "llvm.metadata"

; Function Attrs: convergent mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none)
define ptx_kernel void @control({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, { i32, i32, i64, float, float, float, float } %v1, ptr nofree readonly captures(none) %v2) #0 !dbg !4 {
entry:
  %v1.fca.0.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 0
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !8
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !dbg !13
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !dbg !14
  %v9.i = zext nneg i32 %v2.i to i64, !dbg !15
  %v10.i = zext nneg i32 %v3.i to i64, !dbg !18
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i, !dbg !19
  %v11.i = zext nneg i32 %v4.i to i64, !dbg !20
  %v18.i = add nuw nsw i64 %v17.i, %v11.i, !dbg !19
  %v4.i43 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5, !dbg !21
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #5, !dbg !21
  %v5.i = icmp eq i32 %v4.i43, 1, !dbg !21
  %v7.i = icmp eq i32 %v6.i, 1, !dbg !21
  %v8.not.not.i = and i1 %v5.i, %v7.i, !dbg !25
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5, !dbg !25
  %v14.i = icmp eq i32 %v13.i, 1, !dbg !25
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5, !dbg !25
  %v16.i = icmp eq i32 %v15.i, 1, !dbg !25
  %v17.i44 = and i1 %v14.i, %v16.i, !dbg !25
  %.v18.i = and i1 %v8.not.not.i, %v17.i44, !dbg !27
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1, !dbg !29
  %v17 = zext i32 %v1.fca.0.extract to i64, !dbg !33
  %v18.not = icmp ult i64 %v22.i, %v17, !dbg !35
  br i1 %v18.not, label %bb3, label %bb4, !dbg !35

bb3:                                              ; preds = %entry
  %v0.fca.2.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 2
  %v0.fca.1.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 1
  %v0.fca.0.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 0
  %v22 = getelementptr inbounds nuw [64 x i8], ptr %v0.fca.0.extract, i64 %v22.i, !dbg !36
  %v24.unpack = load float, ptr %v22, align 16, !dbg !44
  %v24.elt1 = getelementptr inbounds nuw i8, ptr %v22, i64 4, !dbg !44
  %v24.unpack2 = load float, ptr %v24.elt1, align 4, !dbg !44
  %v24.elt3 = getelementptr inbounds nuw i8, ptr %v22, i64 8, !dbg !44
  %v24.unpack4 = load float, ptr %v24.elt3, align 8, !dbg !44
  %v25 = getelementptr inbounds nuw i8, ptr %v22, i64 16, !dbg !44
  %v26.unpack = load float, ptr %v25, align 16, !dbg !44
  %v26.elt8 = getelementptr inbounds nuw i8, ptr %v22, i64 20, !dbg !44
  %v26.unpack9 = load float, ptr %v26.elt8, align 4, !dbg !44
  %v26.elt10 = getelementptr inbounds nuw i8, ptr %v22, i64 24, !dbg !44
  %v26.unpack11 = load float, ptr %v26.elt10, align 8, !dbg !44
  %v31 = getelementptr inbounds nuw [8 x i8], ptr %v2, i64 %v22.i, !dbg !45
  %v32 = load i64, ptr %v31, align 8, !dbg !49
  %v33 = sitofp i64 %v32 to float, !dbg !50
  %v34 = fadd nnan contract float %v33, -1.000000e+00, !dbg !51
  %v35 = fmul nnan contract float %v34, 2.000000e+00, !dbg !51
  %v38 = getelementptr inbounds nuw [20 x i8], ptr %v0.fca.2.extract, i64 %v22.i, !dbg !36
  %v46 = fmul nnan contract float %v35, 5.000000e-01, !dbg !52
  store float %v24.unpack, ptr %v38, align 4, !dbg !53
  %v38.repack24 = getelementptr inbounds nuw i8, ptr %v38, i64 4, !dbg !53
  store float %v24.unpack4, ptr %v38.repack24, align 4, !dbg !53
  %v38.repack30 = getelementptr inbounds nuw i8, ptr %v38, i64 16, !dbg !53
  store float %v46, ptr %v38.repack30, align 4, !dbg !53
  %v51 = fsub contract float %v35, %v26.unpack, !dbg !54
  %v52 = fmul contract float %v51, 8.000000e+00, !dbg !55
  %v55 = fsub contract float 1.200000e+00, %v24.unpack2, !dbg !56
  %v56 = fmul contract float %v55, 2.000000e+01, !dbg !57
  %v57 = fadd contract float %v56, 9.810000e+00, !dbg !58
  %v60 = fmul contract float %v26.unpack9, 8.000000e+00, !dbg !59
  %v61 = fsub contract float %v57, %v60, !dbg !58
  %v64 = fsub contract float -2.000000e+00, %v26.unpack11, !dbg !60
  %v65 = fmul contract float %v64, 8.000000e+00, !dbg !61
  %v78 = getelementptr inbounds nuw [32 x i8], ptr %v0.fca.1.extract, i64 %v22.i, !dbg !36
  store float %v52, ptr %v78, align 16, !dbg !62
  %v78.repack34 = getelementptr inbounds nuw i8, ptr %v78, i64 4, !dbg !62
  store float %v61, ptr %v78.repack34, align 4, !dbg !62
  %v78.repack36 = getelementptr inbounds nuw i8, ptr %v78, i64 8, !dbg !62
  store float %v65, ptr %v78.repack36, align 8, !dbg !62
  %v78.repack38 = getelementptr inbounds nuw i8, ptr %v78, i64 12, !dbg !62
  store float 0.000000e+00, ptr %v78.repack38, align 4, !dbg !62
  %v78.repack32 = getelementptr inbounds nuw i8, ptr %v78, i64 16, !dbg !62
  store float 0.000000e+00, ptr %v78.repack32, align 16, !dbg !62
  %v78.repack32.repack40 = getelementptr inbounds nuw i8, ptr %v78, i64 20, !dbg !62
  store float 0.000000e+00, ptr %v78.repack32.repack40, align 4, !dbg !62
  %v78.repack32.repack41 = getelementptr inbounds nuw i8, ptr %v78, i64 24, !dbg !62
  store float 0.000000e+00, ptr %v78.repack32.repack41, align 8, !dbg !62
  %v78.repack32.repack42 = getelementptr inbounds nuw i8, ptr %v78, i64 28, !dbg !62
  store float 0.000000e+00, ptr %v78.repack32.repack42, align 4, !dbg !62
  br label %bb4, !dbg !63

bb4:                                              ; preds = %entry, %bb3
  ret void, !dbg !63
}

; Function Attrs: convergent nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none)
define ptx_kernel void @reset({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, { i32, i32, i64, float, float, float, float } %v1) #1 !dbg !64 {
entry:
  %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i = alloca [36 x float], align 4
  %v1.fca.0.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 0
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !65
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !dbg !68
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !dbg !69
  %v9.i = zext nneg i32 %v2.i to i64, !dbg !70
  %v10.i = zext nneg i32 %v3.i to i64, !dbg !72
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i, !dbg !73
  %v11.i = zext nneg i32 %v4.i to i64, !dbg !74
  %v18.i = add nuw nsw i64 %v17.i, %v11.i, !dbg !73
  %v4.i12 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5, !dbg !75
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #5, !dbg !75
  %v5.i = icmp eq i32 %v4.i12, 1, !dbg !75
  %v7.i = icmp eq i32 %v6.i, 1, !dbg !75
  %v8.not.not.i = and i1 %v5.i, %v7.i, !dbg !77
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5, !dbg !77
  %v14.i = icmp eq i32 %v13.i, 1, !dbg !77
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5, !dbg !77
  %v16.i = icmp eq i32 %v15.i, 1, !dbg !77
  %v17.i13 = and i1 %v14.i, %v16.i, !dbg !77
  %.v18.i = and i1 %v8.not.not.i, %v17.i13, !dbg !78
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1, !dbg !79
  %v11 = zext i32 %v1.fca.0.extract to i64, !dbg !81
  %v12.not = icmp ult i64 %v22.i, %v11, !dbg !83
  br i1 %v12.not, label %bb3, label %bb6, !dbg !83

bb3:                                              ; preds = %entry
  %v1.fca.6.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 6
  %v1.fca.5.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 5
  %v1.fca.4.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 4
  %v1.fca.3.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 3
  %v1.fca.2.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 2
  %v1.fca.1.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 1
  %v0.fca.10.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 10
  %v0.fca.9.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 9
  %v0.fca.8.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 8
  %v0.fca.7.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 7
  %v0.fca.6.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 6
  %v0.fca.5.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 5
  %v0.fca.4.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 4
  %v0.fca.3.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 3
  %v0.fca.2.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 2
  %v0.fca.1.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 1
  %v0.fca.0.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 0
  %v16 = getelementptr inbounds nuw [8 x i8], ptr %v0.fca.10.extract, i64 %v22.i, !dbg !84
  store i64 0, ptr %v16, align 8, !dbg !87
  %v25 = getelementptr inbounds nuw [32 x i8], ptr %v0.fca.1.extract, i64 %v22.i, !dbg !84
  store float 0.000000e+00, ptr %v25, align 16, !dbg !88
  %v25.repack6 = getelementptr inbounds nuw i8, ptr %v25, i64 4, !dbg !88
  store float 0.000000e+00, ptr %v25.repack6, align 4, !dbg !88
  %v25.repack7 = getelementptr inbounds nuw i8, ptr %v25, i64 8, !dbg !88
  store float 0.000000e+00, ptr %v25.repack7, align 8, !dbg !88
  %v25.repack8 = getelementptr inbounds nuw i8, ptr %v25, i64 12, !dbg !88
  store float 0.000000e+00, ptr %v25.repack8, align 4, !dbg !88
  %v25.repack5 = getelementptr inbounds nuw i8, ptr %v25, i64 16, !dbg !88
  store float 0.000000e+00, ptr %v25.repack5, align 16, !dbg !88
  %v25.repack5.repack9 = getelementptr inbounds nuw i8, ptr %v25, i64 20, !dbg !88
  store float 0.000000e+00, ptr %v25.repack5.repack9, align 4, !dbg !88
  %v25.repack5.repack10 = getelementptr inbounds nuw i8, ptr %v25, i64 24, !dbg !88
  store float 0.000000e+00, ptr %v25.repack5.repack10, align 8, !dbg !88
  %v25.repack5.repack11 = getelementptr inbounds nuw i8, ptr %v25, i64 28, !dbg !88
  store float 0.000000e+00, ptr %v25.repack5.repack11, align 4, !dbg !88
  %v28 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.5.extract, i64 %v22.i, !dbg !84
  store float 0.000000e+00, ptr %v28, align 4, !dbg !89
  %v31 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.6.extract, i64 %v22.i, !dbg !84
  store float 0.000000e+00, ptr %v31, align 4, !dbg !90
  %v34 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.7.extract, i64 %v22.i, !dbg !84
  store float 0.000000e+00, ptr %v34, align 4, !dbg !91
  %v37 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.9.extract, i64 %v22.i, !dbg !84
  store float 0.000000e+00, ptr %v37, align 4, !dbg !92
  %v48.i = uitofp nneg i64 %v22.i to float, !dbg !93
  %v49.i = fmul nnan contract float %v48.i, 3.200000e+01, !dbg !93
  %v54.i = mul i64 %v22.i, -7046029254386353131, !dbg !96
  %v55.i = xor i64 %v54.i, %v1.fca.2.extract, !dbg !106
  %v58.i = zext i32 %v1.fca.1.extract to i64, !dbg !107
  %v59.i = mul i64 %v58.i, -7723592293110705685, !dbg !96
  %v57.i = xor i64 %v55.i, %v59.i, !dbg !106
  %v63.i = lshr i64 %v57.i, 30, !dbg !108
  %v64.i = xor i64 %v63.i, %v57.i, !dbg !109
  %v65.i = mul i64 %v64.i, -4658895280553007687, !dbg !96
  %v68.i = lshr i64 %v65.i, 27, !dbg !110
  %v69.i = xor i64 %v68.i, %v65.i, !dbg !111
  %v70.i = mul i64 %v69.i, -7723592293110705685, !dbg !96
  %v77.i = lshr i64 %v70.i, 40, !dbg !112
  %v78.i = uitofp nneg i64 %v77.i to float, !dbg !112
  %v79.i = fmul nnan contract float %v78.i, f0x34A00000, !dbg !112
  %v80.i = fadd contract float %v79.i, -2.500000e+00, !dbg !112
  %v102.i = getelementptr inbounds nuw [64 x i8], ptr %v0.fca.0.extract, i64 %v22.i, !dbg !113
  store float %v49.i, ptr %v102.i, align 16, !dbg !119
  %v102.repack7.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 4, !dbg !119
  store float 1.200000e+00, ptr %v102.repack7.i, align 4, !dbg !119
  %v102.repack9.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 8, !dbg !119
  store float 0.000000e+00, ptr %v102.repack9.i, align 8, !dbg !119
  %v102.repack11.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 12, !dbg !119
  store float 0.000000e+00, ptr %v102.repack11.i, align 4, !dbg !119
  %v102.repack1.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 16, !dbg !119
  store float 0.000000e+00, ptr %v102.repack1.i, align 16, !dbg !119
  %v102.repack1.repack13.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 20, !dbg !119
  store float 0.000000e+00, ptr %v102.repack1.repack13.i, align 4, !dbg !119
  %v102.repack1.repack14.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 24, !dbg !119
  store float -2.000000e+00, ptr %v102.repack1.repack14.i, align 8, !dbg !119
  %v102.repack1.repack15.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 28, !dbg !119
  store float 0.000000e+00, ptr %v102.repack1.repack15.i, align 4, !dbg !119
  %v102.repack3.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 32, !dbg !119
  store float 0.000000e+00, ptr %v102.repack3.i, align 16, !dbg !119
  %v102.repack3.repack16.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 36, !dbg !119
  store float 0.000000e+00, ptr %v102.repack3.repack16.i, align 4, !dbg !119
  %v102.repack3.repack17.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 40, !dbg !119
  store float 0.000000e+00, ptr %v102.repack3.repack17.i, align 8, !dbg !119
  %v102.repack3.repack18.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 44, !dbg !119
  store float 1.000000e+00, ptr %v102.repack3.repack18.i, align 4, !dbg !119
  %v102.repack5.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 48, !dbg !119
  store float 0.000000e+00, ptr %v102.repack5.i, align 16, !dbg !119
  %v102.repack5.repack19.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 52, !dbg !119
  store float 0.000000e+00, ptr %v102.repack5.repack19.i, align 4, !dbg !119
  %v102.repack5.repack20.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 56, !dbg !119
  store float 0.000000e+00, ptr %v102.repack5.repack20.i, align 8, !dbg !119
  %v102.repack5.repack21.i = getelementptr inbounds nuw i8, ptr %v102.i, i64 60, !dbg !119
  store float 0.000000e+00, ptr %v102.repack5.repack21.i, align 4, !dbg !119
  %v109.i = getelementptr inbounds nuw [20 x i8], ptr %v0.fca.2.extract, i64 %v22.i, !dbg !113
  store float %v49.i, ptr %v109.i, align 4, !dbg !120
  %v109.repack22.i = getelementptr inbounds nuw i8, ptr %v109.i, i64 4, !dbg !120
  store float 0.000000e+00, ptr %v109.repack22.i, align 4, !dbg !120
  %v109.repack24.i = getelementptr inbounds nuw i8, ptr %v109.i, i64 8, !dbg !120
  store float %v80.i, ptr %v109.repack24.i, align 4, !dbg !120
  %v109.repack26.i = getelementptr inbounds nuw i8, ptr %v109.i, i64 12, !dbg !120
  store i32 0, ptr %v109.repack26.i, align 4, !dbg !120
  %v109.repack28.i = getelementptr inbounds nuw i8, ptr %v109.i, i64 16, !dbg !120
  store float 0.000000e+00, ptr %v109.repack28.i, align 4, !dbg !120
  %v110.i = fadd contract float %v49.i, %v80.i, !dbg !121
  %v111.i = fadd contract float %v110.i, -8.000000e-01, !dbg !121
  %v112.i = fadd contract float %v111.i, -4.000000e+00, !dbg !121
  %.idx.i = shl nuw nsw i64 %v22.i, 7, !dbg !113
  %0 = getelementptr i8, ptr %v0.fca.0.extract, i64 %.idx.i, !dbg !113
  %v125.i = getelementptr [64 x i8], ptr %0, i64 %v11, !dbg !113
  store float %v112.i, ptr %v125.i, align 16, !dbg !122
  %v125.repack36.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 4, !dbg !122
  store float 1.500000e+00, ptr %v125.repack36.i, align 4, !dbg !122
  %v125.repack38.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 8, !dbg !122
  store float -6.000000e+00, ptr %v125.repack38.i, align 8, !dbg !122
  %v125.repack40.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 12, !dbg !122
  store float 0.000000e+00, ptr %v125.repack40.i, align 4, !dbg !122
  %v125.repack30.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 16, !dbg !122
  store float 0.000000e+00, ptr %v125.repack30.i, align 16, !dbg !122
  %v125.repack30.repack42.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 20, !dbg !122
  store float 0.000000e+00, ptr %v125.repack30.repack42.i, align 4, !dbg !122
  %v125.repack30.repack43.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 24, !dbg !122
  store float 0.000000e+00, ptr %v125.repack30.repack43.i, align 8, !dbg !122
  %v125.repack30.repack44.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 28, !dbg !122
  store float 0.000000e+00, ptr %v125.repack30.repack44.i, align 4, !dbg !122
  %v125.repack32.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 32, !dbg !122
  store float 0.000000e+00, ptr %v125.repack32.i, align 16, !dbg !122
  %v125.repack32.repack45.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 36, !dbg !122
  store float 0.000000e+00, ptr %v125.repack32.repack45.i, align 4, !dbg !122
  %v125.repack32.repack46.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 40, !dbg !122
  store float 0.000000e+00, ptr %v125.repack32.repack46.i, align 8, !dbg !122
  %v125.repack32.repack47.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 44, !dbg !122
  store float 1.000000e+00, ptr %v125.repack32.repack47.i, align 4, !dbg !122
  %v125.repack34.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 48, !dbg !122
  store float 0.000000e+00, ptr %v125.repack34.i, align 16, !dbg !122
  %v125.repack34.repack48.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 52, !dbg !122
  store float 0.000000e+00, ptr %v125.repack34.repack48.i, align 4, !dbg !122
  %v125.repack34.repack49.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 56, !dbg !122
  store float 0.000000e+00, ptr %v125.repack34.repack49.i, align 8, !dbg !122
  %v125.repack34.repack50.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 60, !dbg !122
  store float 0.000000e+00, ptr %v125.repack34.repack50.i, align 4, !dbg !122
  %v127.i = fadd contract float %v110.i, 8.000000e-01, !dbg !123
  %v128.i = fadd contract float %v127.i, 4.000000e+00, !dbg !124
  %v132.i = getelementptr i8, ptr %v125.i, i64 64, !dbg !113
  store float %v128.i, ptr %v132.i, align 16, !dbg !125
  %v132.repack57.i = getelementptr i8, ptr %v125.i, i64 68, !dbg !125
  store float 1.500000e+00, ptr %v132.repack57.i, align 4, !dbg !125
  %v132.repack59.i = getelementptr i8, ptr %v125.i, i64 72, !dbg !125
  store float -6.000000e+00, ptr %v132.repack59.i, align 8, !dbg !125
  %v132.repack61.i = getelementptr i8, ptr %v125.i, i64 76, !dbg !125
  store float 0.000000e+00, ptr %v132.repack61.i, align 4, !dbg !125
  %v132.repack51.i = getelementptr i8, ptr %v125.i, i64 80, !dbg !125
  store float 0.000000e+00, ptr %v132.repack51.i, align 16, !dbg !125
  %v132.repack51.repack63.i = getelementptr i8, ptr %v125.i, i64 84, !dbg !125
  store float 0.000000e+00, ptr %v132.repack51.repack63.i, align 4, !dbg !125
  %v132.repack51.repack64.i = getelementptr i8, ptr %v125.i, i64 88, !dbg !125
  store float 0.000000e+00, ptr %v132.repack51.repack64.i, align 8, !dbg !125
  %v132.repack51.repack65.i = getelementptr i8, ptr %v125.i, i64 92, !dbg !125
  store float 0.000000e+00, ptr %v132.repack51.repack65.i, align 4, !dbg !125
  %v132.repack53.i = getelementptr i8, ptr %v125.i, i64 96, !dbg !125
  store float 0.000000e+00, ptr %v132.repack53.i, align 16, !dbg !125
  %v132.repack53.repack66.i = getelementptr i8, ptr %v125.i, i64 100, !dbg !125
  store float 0.000000e+00, ptr %v132.repack53.repack66.i, align 4, !dbg !125
  %v132.repack53.repack67.i = getelementptr i8, ptr %v125.i, i64 104, !dbg !125
  store float 0.000000e+00, ptr %v132.repack53.repack67.i, align 8, !dbg !125
  %v132.repack53.repack68.i = getelementptr i8, ptr %v125.i, i64 108, !dbg !125
  store float 1.000000e+00, ptr %v132.repack53.repack68.i, align 4, !dbg !125
  %v132.repack55.i = getelementptr i8, ptr %v125.i, i64 112, !dbg !125
  store float 0.000000e+00, ptr %v132.repack55.i, align 16, !dbg !125
  %v132.repack55.repack69.i = getelementptr i8, ptr %v125.i, i64 116, !dbg !125
  store float 0.000000e+00, ptr %v132.repack55.repack69.i, align 4, !dbg !125
  %v132.repack55.repack70.i = getelementptr i8, ptr %v125.i, i64 120, !dbg !125
  store float 0.000000e+00, ptr %v132.repack55.repack70.i, align 8, !dbg !125
  %v132.repack55.repack71.i = getelementptr i8, ptr %v125.i, i64 124, !dbg !125
  store float 0.000000e+00, ptr %v132.repack55.repack71.i, align 4, !dbg !125
  call void @llvm.lifetime.start.p0(ptr nonnull %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i), !dbg !126
  %v51.unpack.i = load float, ptr %v102.i, align 16, !dbg !143
  %v51.unpack2.i = load float, ptr %v102.repack7.i, align 4, !dbg !143
  %v51.unpack4.i = load float, ptr %v102.repack9.i, align 8, !dbg !143
  %v53.unpack.i = load float, ptr %v102.repack1.i, align 16, !dbg !143
  %v53.unpack9.i = load float, ptr %v102.repack1.repack13.i, align 4, !dbg !143
  %v53.unpack11.i = load float, ptr %v102.repack1.repack14.i, align 8, !dbg !143
  %v66.i = load float, ptr %v109.repack24.i, align 4, !dbg !144
  %v68.i16 = load i32, ptr %v109.repack26.i, align 4, !dbg !144
  %v70.i18 = load float, ptr %v109.repack28.i, align 4, !dbg !144
  %v79.i21 = uitofp i32 %v68.i16 to float, !dbg !145
  %v80.i22 = fdiv contract float %v79.i21, 1.500000e+02, !dbg !145
  %v81.i = fsub contract float %v49.i, %v51.unpack.i, !dbg !146
  %v82.i = fsub contract float 1.200000e+00, %v51.unpack2.i, !dbg !147
  %v83.i = fsub contract float -1.200000e+01, %v51.unpack4.i, !dbg !148
  %v90.i = fsub contract float 1.000000e+00, %v80.i22, !dbg !149
  %v99.i = fadd contract float %v49.i, %v66.i, !dbg !150
  %v100.i = fsub contract float %v99.i, %v51.unpack.i, !dbg !150
  %v101.i = fsub contract float -6.000000e+00, %v51.unpack4.i, !dbg !151
  %.idx.i23 = shl nuw nsw i64 %v22.i, 5
  %1 = getelementptr i8, ptr %v0.fca.4.extract, i64 %.idx.i23
  store float %v81.i, ptr %1, align 4, !dbg !152
  %v116.1.i = getelementptr i8, ptr %1, i64 4, !dbg !126
  store float %v82.i, ptr %v116.1.i, align 4, !dbg !152
  %v116.2.i = getelementptr i8, ptr %1, i64 8, !dbg !126
  store float %v83.i, ptr %v116.2.i, align 4, !dbg !152
  %v116.3.i = getelementptr i8, ptr %1, i64 12, !dbg !126
  store float %v53.unpack.i, ptr %v116.3.i, align 4, !dbg !152
  %v116.4.i = getelementptr i8, ptr %1, i64 16, !dbg !126
  store float %v53.unpack9.i, ptr %v116.4.i, align 4, !dbg !152
  %v116.5.i = getelementptr i8, ptr %1, i64 20, !dbg !126
  store float %v53.unpack11.i, ptr %v116.5.i, align 4, !dbg !152
  %v116.6.i = getelementptr i8, ptr %1, i64 24, !dbg !126
  store float %v90.i, ptr %v116.6.i, align 4, !dbg !152
  %v116.7.i = getelementptr i8, ptr %1, i64 28, !dbg !126
  store float %v70.i18, ptr %v116.7.i, align 4, !dbg !152
  %.idx15.i = shl nuw nsw i64 %v22.i, 4
  %2 = getelementptr i8, ptr %v0.fca.8.extract, i64 %.idx15.i
  store float %v100.i, ptr %2, align 4, !dbg !153
  %v128.1.i = getelementptr i8, ptr %2, i64 4, !dbg !126
  store float %v101.i, ptr %v128.1.i, align 4, !dbg !153
  %v128.2.i = getelementptr i8, ptr %2, i64 8, !dbg !126
  store float 8.000000e-01, ptr %v128.2.i, align 4, !dbg !153
  %v128.3.i = getelementptr i8, ptr %2, i64 12, !dbg !126
  store float %v80.i22, ptr %v128.3.i, align 4, !dbg !153
  %v138.i = fneg float %v51.unpack.i, !dbg !154
  %v139.i = fneg float %v51.unpack2.i, !dbg !155
  %v140.i = fneg float %v51.unpack4.i, !dbg !156
  %v141.i = fneg float %v1.fca.3.extract, !dbg !157
  %v142.i = fmul contract float %v51.unpack.i, %v141.i, !dbg !157
  %v143.i = fneg float %v1.fca.4.extract, !dbg !158
  %v144.i = fmul contract float %v51.unpack2.i, %v143.i, !dbg !158
  %v145.i = fmul contract float %v1.fca.5.extract, %v51.unpack4.i, !dbg !159
  %v146.i = fsub contract float %v1.fca.6.extract, %v145.i, !dbg !160
  store float 1.000000e+00, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, align 4, !dbg !161
  %v182.fca.1.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 4, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.1.gep.i, align 4, !dbg !161
  %v182.fca.2.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 8, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.2.gep.i, align 4, !dbg !161
  %v182.fca.3.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 12, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.3.gep.i, align 4, !dbg !161
  %v182.fca.4.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 16, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.4.gep.i, align 4, !dbg !161
  %v182.fca.5.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 20, !dbg !161
  store float 1.000000e+00, ptr %v182.fca.5.gep.i, align 4, !dbg !161
  %v182.fca.6.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 24, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.6.gep.i, align 4, !dbg !161
  %v182.fca.7.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 28, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.7.gep.i, align 4, !dbg !161
  %v182.fca.8.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 32, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.8.gep.i, align 4, !dbg !161
  %v182.fca.9.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 36, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.9.gep.i, align 4, !dbg !161
  %v182.fca.10.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 40, !dbg !161
  store float 1.000000e+00, ptr %v182.fca.10.gep.i, align 4, !dbg !161
  %v182.fca.11.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 44, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.11.gep.i, align 4, !dbg !161
  %v182.fca.12.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 48, !dbg !161
  store float %v138.i, ptr %v182.fca.12.gep.i, align 4, !dbg !161
  %v182.fca.13.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 52, !dbg !161
  store float %v139.i, ptr %v182.fca.13.gep.i, align 4, !dbg !161
  %v182.fca.14.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 56, !dbg !161
  store float %v140.i, ptr %v182.fca.14.gep.i, align 4, !dbg !161
  %v182.fca.15.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 60, !dbg !161
  store float 1.000000e+00, ptr %v182.fca.15.gep.i, align 4, !dbg !161
  %v182.fca.16.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 64, !dbg !161
  store float %v1.fca.3.extract, ptr %v182.fca.16.gep.i, align 4, !dbg !161
  %v182.fca.17.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 68, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.17.gep.i, align 4, !dbg !161
  %v182.fca.18.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 72, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.18.gep.i, align 4, !dbg !161
  %v182.fca.19.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 76, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.19.gep.i, align 4, !dbg !161
  %v182.fca.20.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 80, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.20.gep.i, align 4, !dbg !161
  %v182.fca.21.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 84, !dbg !161
  store float %v1.fca.4.extract, ptr %v182.fca.21.gep.i, align 4, !dbg !161
  %v182.fca.22.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 88, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.22.gep.i, align 4, !dbg !161
  %v182.fca.23.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 92, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.23.gep.i, align 4, !dbg !161
  %v182.fca.24.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 96, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.24.gep.i, align 4, !dbg !161
  %v182.fca.25.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 100, !dbg !161
  store float 0.000000e+00, ptr %v182.fca.25.gep.i, align 4, !dbg !161
  %v182.fca.26.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 104, !dbg !161
  store float %v1.fca.5.extract, ptr %v182.fca.26.gep.i, align 4, !dbg !161
  %v182.fca.27.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 108, !dbg !161
  store float -1.000000e+00, ptr %v182.fca.27.gep.i, align 4, !dbg !161
  %v182.fca.28.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 112, !dbg !161
  store float %v142.i, ptr %v182.fca.28.gep.i, align 4, !dbg !161
  %v182.fca.29.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 116, !dbg !161
  store float %v144.i, ptr %v182.fca.29.gep.i, align 4, !dbg !161
  %v182.fca.30.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 120, !dbg !161
  store float %v146.i, ptr %v182.fca.30.gep.i, align 4, !dbg !161
  %v182.fca.31.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 124, !dbg !161
  store float %v51.unpack4.i, ptr %v182.fca.31.gep.i, align 4, !dbg !161
  %v182.fca.32.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 128, !dbg !161
  store float %v51.unpack.i, ptr %v182.fca.32.gep.i, align 4, !dbg !161
  %v182.fca.33.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 132, !dbg !161
  store float %v51.unpack2.i, ptr %v182.fca.33.gep.i, align 4, !dbg !161
  %v182.fca.34.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 136, !dbg !161
  store float %v51.unpack4.i, ptr %v182.fca.34.gep.i, align 4, !dbg !161
  %v182.fca.35.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 140, !dbg !161
  store float 1.000000e+00, ptr %v182.fca.35.gep.i, align 4, !dbg !161
  %.idx16.i = mul nuw nsw i64 %v22.i, 144
  %3 = getelementptr i8, ptr %v0.fca.3.extract, i64 %.idx16.i
  br label %bb14.i, !dbg !162

bb14.i:                                           ; preds = %bb14.i, %bb3
  %v18319.i = phi i64 [ 0, %bb3 ], [ %v194.i.3, %bb14.i ]
  %v187.i = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v18319.i, !dbg !163
  %v188.i = load float, ptr %v187.i, align 4, !dbg !163
  %v193.i = getelementptr [4 x i8], ptr %3, i64 %v18319.i, !dbg !126
  store float %v188.i, ptr %v193.i, align 4, !dbg !164
  %v194.i = or disjoint i64 %v18319.i, 1, !dbg !165
  %v187.i.1 = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v194.i, !dbg !163
  %v188.i.1 = load float, ptr %v187.i.1, align 4, !dbg !163
  %v193.i.1 = getelementptr [4 x i8], ptr %3, i64 %v194.i, !dbg !126
  store float %v188.i.1, ptr %v193.i.1, align 4, !dbg !164
  %v194.i.1 = or disjoint i64 %v18319.i, 2, !dbg !165
  %v187.i.2 = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v194.i.1, !dbg !163
  %v188.i.2 = load float, ptr %v187.i.2, align 4, !dbg !163
  %v193.i.2 = getelementptr [4 x i8], ptr %3, i64 %v194.i.1, !dbg !126
  store float %v188.i.2, ptr %v193.i.2, align 4, !dbg !164
  %v194.i.2 = or disjoint i64 %v18319.i, 3, !dbg !165
  %v187.i.3 = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v194.i.2, !dbg !163
  %v188.i.3 = load float, ptr %v187.i.3, align 4, !dbg !163
  %v193.i.3 = getelementptr [4 x i8], ptr %3, i64 %v194.i.2, !dbg !126
  store float %v188.i.3, ptr %v193.i.3, align 4, !dbg !164
  %v194.i.3 = add nuw nsw i64 %v18319.i, 4, !dbg !165
  %exitcond.i.3 = icmp eq i64 %v194.i.3, 36, !dbg !162
  br i1 %exitcond.i.3, label %sim_env__kernels__device__observe_one.exit, label %bb14.i, !dbg !162

sim_env__kernels__device__observe_one.exit:       ; preds = %bb14.i
  call void @llvm.lifetime.end.p0(ptr nonnull %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i), !dbg !166
  br label %bb6, !dbg !167

bb6:                                              ; preds = %entry, %sim_env__kernels__device__observe_one.exit
  ret void, !dbg !167
}

; Function Attrs: convergent nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none)
define ptx_kernel void @finish({ ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, { i32, i32, i64, float, float, float, float } %v1) #1 !dbg !168 {
entry:
  %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i = alloca [36 x float], align 4
  %v0.fca.0.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 0
  %v0.fca.3.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 3
  %v0.fca.4.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 4
  %v0.fca.5.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 5
  %v0.fca.6.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 6
  %v0.fca.7.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 7
  %v0.fca.8.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 8
  %v0.fca.9.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 9
  %v0.fca.10.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 10
  %v1.fca.0.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 0
  %v1.fca.1.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 1
  %v1.fca.2.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 2
  %v1.fca.3.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 3
  %v1.fca.4.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 4
  %v1.fca.5.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 5
  %v1.fca.6.extract = extractvalue { i32, i32, i64, float, float, float, float } %v1, 6
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #5, !dbg !169
  %v3.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5, !dbg !172
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5, !dbg !173
  %v9.i = zext nneg i32 %v2.i to i64, !dbg !174
  %v10.i = zext nneg i32 %v3.i to i64, !dbg !176
  %v17.i = mul nuw nsw i64 %v9.i, %v10.i, !dbg !177
  %v11.i = zext nneg i32 %v4.i to i64, !dbg !178
  %v18.i = add nuw nsw i64 %v17.i, %v11.i, !dbg !177
  %v4.i32 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #5, !dbg !179
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #5, !dbg !179
  %v5.i = icmp eq i32 %v4.i32, 1, !dbg !179
  %v7.i = icmp eq i32 %v6.i, 1, !dbg !179
  %v8.not.not.i = and i1 %v5.i, %v7.i, !dbg !181
  %v13.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #5, !dbg !181
  %v14.i = icmp eq i32 %v13.i, 1, !dbg !181
  %v15.i = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #5, !dbg !181
  %v16.i = icmp eq i32 %v15.i, 1, !dbg !181
  %v17.i33 = and i1 %v14.i, %v16.i, !dbg !181
  %.v18.i = and i1 %v8.not.not.i, %v17.i33, !dbg !182
  %v22.i = select i1 %.v18.i, i64 %v18.i, i64 -1, !dbg !183
  %v12 = zext i32 %v1.fca.0.extract to i64, !dbg !185
  %v13.not = icmp ult i64 %v22.i, %v12, !dbg !187
  br i1 %v13.not, label %bb3, label %bb35, !dbg !187

bb3:                                              ; preds = %entry
  %v0.fca.2.extract = extractvalue { ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr, ptr } %v0, 2
  %v17 = getelementptr inbounds nuw [64 x i8], ptr %v0.fca.0.extract, i64 %v22.i, !dbg !188
  %v19.unpack = load float, ptr %v17, align 16, !dbg !198
  %v19.elt10 = getelementptr inbounds nuw i8, ptr %v17, i64 8, !dbg !198
  %v19.unpack11 = load float, ptr %v19.elt10, align 8, !dbg !198
  %v28 = getelementptr inbounds nuw [20 x i8], ptr %v0.fca.2.extract, i64 %v22.i, !dbg !188
  %v29.unpack = load float, ptr %v28, align 4, !dbg !199
  %v29.elt15 = getelementptr inbounds nuw i8, ptr %v28, i64 4, !dbg !199
  %v29.unpack16 = load float, ptr %v29.elt15, align 4, !dbg !199
  %v29.elt17 = getelementptr inbounds nuw i8, ptr %v28, i64 8, !dbg !199
  %v29.unpack18 = load float, ptr %v29.elt17, align 4, !dbg !199
  %v29.elt19 = getelementptr inbounds nuw i8, ptr %v28, i64 12, !dbg !199
  %v29.unpack20 = load i32, ptr %v29.elt19, align 4, !dbg !199
  %v29.elt21 = getelementptr inbounds nuw i8, ptr %v28, i64 16, !dbg !199
  %v29.unpack22 = load float, ptr %v29.elt21, align 4, !dbg !199
  %v32 = uitofp nneg i64 %v22.i to float, !dbg !200
  %v33 = fmul nnan contract float %v32, 3.200000e+01, !dbg !200
  %v34 = fsub contract float %v19.unpack, %v33, !dbg !201
  %0 = tail call float @llvm.fabs.f32(float %v34), !dbg !202
  %or.cond = fcmp ule float %0, 3.820000e+00, !dbg !202
  br i1 %or.cond, label %bb6, label %bb8, !dbg !202

bb6:                                              ; preds = %bb3
  %v41 = fsub contract float %v29.unpack, %v33, !dbg !203
  %v10.i34 = fcmp ule float %v29.unpack16, -5.670000e+00, !dbg !204
  %v12.i = fcmp ule float %v19.unpack11, -5.670000e+00
  %or.cond.i = select i1 %v10.i34, i1 true, i1 %v12.i, !dbg !204
  br i1 %or.cond.i, label %bb2.i, label %bb8, !dbg !204

bb2.i:                                            ; preds = %bb6
  %v14.i35 = fcmp uge float %v29.unpack16, -6.330000e+00, !dbg !209
  %v16.i36 = fcmp uge float %v19.unpack11, -6.330000e+00
  %or.cond1.i = select i1 %v14.i35, i1 true, i1 %v16.i36, !dbg !209
  br i1 %or.cond1.i, label %bb5.i, label %bb8, !dbg !209

bb5.i:                                            ; preds = %bb2.i
  %v18.i37 = fsub contract float %v19.unpack11, %v29.unpack16, !dbg !210
  %v19.i = fcmp oeq float %v18.i37, 0.000000e+00, !dbg !211
  br i1 %v19.i, label %bb16.i, label %bb6.i, !dbg !211

bb6.i:                                            ; preds = %bb5.i
  %v21.i = fsub contract float -5.670000e+00, %v29.unpack16, !dbg !214
  %v22.i38 = fdiv contract float %v21.i, %v18.i37, !dbg !214
  %v23.i = fsub contract float -6.330000e+00, %v29.unpack16, !dbg !215
  %v24.i = fdiv contract float %v23.i, %v18.i37, !dbg !215
  %v25.i = fcmp uge float %v22.i38, %v24.i, !dbg !217
  %v22.v24.i = select i1 %v25.i, float %v22.i38, float %v24.i, !dbg !219
  %v24.v22.i = select i1 %v25.i, float %v24.i, float %v22.i38, !dbg !219
  %v29.inv.i = fcmp ogt float %v24.v22.i, 0.000000e+00, !dbg !220
  %v31.i = select i1 %v29.inv.i, float %v24.v22.i, float 0.000000e+00, !dbg !220
  %v32.inv.i = fcmp olt float %v22.v24.i, 1.000000e+00, !dbg !222
  %v34.i = select i1 %v32.inv.i, float %v22.v24.i, float 1.000000e+00, !dbg !222
  br label %bb16.i, !dbg !223

bb16.i:                                           ; preds = %bb6.i, %bb5.i
  %v35.i = phi float [ 0.000000e+00, %bb5.i ], [ %v31.i, %bb6.i ]
  %v36.i = phi float [ 1.000000e+00, %bb5.i ], [ %v34.i, %bb6.i ]
  %v37.i = fcmp ule float %v35.i, %v36.i, !dbg !224
  br i1 %v37.i, label %bb18.i, label %bb8, !dbg !224

bb18.i:                                           ; preds = %bb16.i
  %v39.i = fsub contract float %v34, %v41, !dbg !225
  %v40.i = fmul contract float %v39.i, %v35.i, !dbg !225
  %v41.i = fadd contract float %v41, %v40.i, !dbg !226
  %v42.i = fsub contract float %v41.i, %v29.unpack18, !dbg !226
  %v43.i = fmul contract float %v39.i, %v36.i, !dbg !227
  %v44.i = fadd contract float %v41, %v43.i, !dbg !229
  %v45.i = fsub contract float %v44.i, %v29.unpack18, !dbg !229
  %1 = tail call float @llvm.fabs.f32(float %v42.i), !dbg !230
  %or.cond2.i = fcmp ule float %1, 6.200000e-01, !dbg !230
  %2 = tail call float @llvm.fabs.f32(float %v45.i), !dbg !230
  %3 = fcmp ule float %2, 6.200000e-01, !dbg !230
  %v53.i.not = select i1 %or.cond2.i, i1 %3, i1 false, !dbg !233
  br label %bb8, !dbg !234

bb8:                                              ; preds = %bb18.i, %bb16.i, %bb2.i, %bb6, %bb3
  %v49 = phi i1 [ false, %bb3 ], [ %v53.i.not, %bb18.i ], [ true, %bb6 ], [ true, %bb2.i ], [ true, %bb16.i ]
  %v53 = fcmp ugt float %v19.unpack11, -1.200000e+01, !dbg !233
  %not.v49 = xor i1 %v49, true, !dbg !233
  %v54 = select i1 %not.v49, i1 true, i1 %v53, !dbg !233
  %v57 = add i32 %v29.unpack20, 1, !dbg !235
  %or.cond1 = select i1 %v49, i1 %v53, i1 false, !dbg !236
  %v63 = icmp ult i32 %v57, 150, !dbg !236
  %v64.not = select i1 %or.cond1, i1 %v63, i1 false, !dbg !237
  %v69 = fsub contract float %v29.unpack16, %v19.unpack11, !dbg !237
  %v70 = fmul contract float %v69, 2.000000e-01, !dbg !237
  %v71 = fadd contract float %v70, f0xBC23D70A, !dbg !237
  %. = select i1 %v53, float 0.000000e+00, float 5.000000e+00, !dbg !238
  %v75 = select i1 %v49, float %., float -5.000000e+00, !dbg !238
  %v76 = fadd contract float %v71, %v75, !dbg !237
  %v79 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.5.extract, i64 %v22.i, !dbg !188
  store float %v76, ptr %v79, align 4, !dbg !239
  %v81 = select i1 %v64.not, float 0.000000e+00, float 1.000000e+00, !dbg !240
  %v84 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.6.extract, i64 %v22.i, !dbg !188
  store float %v81, ptr %v84, align 4, !dbg !241
  %v86 = select i1 %v54, float 0.000000e+00, float 1.000000e+00, !dbg !242
  %v89 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.7.extract, i64 %v22.i, !dbg !188
  store float %v86, ptr %v89, align 4, !dbg !243
  %v91 = select i1 %v49, float 0.000000e+00, float 1.000000e+00, !dbg !244
  %v94 = getelementptr inbounds nuw [4 x i8], ptr %v0.fca.9.extract, i64 %v22.i, !dbg !188
  store float %v91, ptr %v94, align 4, !dbg !245
  store float %v29.unpack, ptr %v28, align 4, !dbg !246
  store float %v29.unpack16, ptr %v29.elt15, align 4, !dbg !246
  store float %v29.unpack18, ptr %v29.elt17, align 4, !dbg !246
  store i32 %v57, ptr %v29.elt19, align 4, !dbg !246
  store float %v29.unpack22, ptr %v29.elt21, align 4, !dbg !246
  br i1 %v64.not, label %bb33, label %bb31, !dbg !247

bb31:                                             ; preds = %bb8
  %v99 = getelementptr inbounds nuw [8 x i8], ptr %v0.fca.10.extract, i64 %v22.i, !dbg !188
  %v100 = load i64, ptr %v99, align 8, !dbg !248
  %v101 = add i64 %v100, 1, !dbg !249
  store i64 %v101, ptr %v99, align 8, !dbg !253
  %v54.i = mul i64 %v22.i, -7046029254386353131, !dbg !255
  %v55.i40 = xor i64 %v54.i, %v1.fca.2.extract, !dbg !259
  %v56.i = mul i64 %v101, -3335678366873096957, !dbg !255
  %v58.i = zext i32 %v1.fca.1.extract to i64, !dbg !260
  %v59.i = mul i64 %v58.i, -7723592293110705685, !dbg !255
  %v57.i = xor i64 %v55.i40, %v59.i, !dbg !259
  %v60.i = xor i64 %v57.i, %v56.i, !dbg !259
  %v63.i = lshr i64 %v60.i, 30, !dbg !261
  %v64.i = xor i64 %v63.i, %v60.i, !dbg !262
  %v65.i = mul i64 %v64.i, -4658895280553007687, !dbg !255
  %v68.i = lshr i64 %v65.i, 27, !dbg !263
  %v69.i = xor i64 %v68.i, %v65.i, !dbg !264
  %v70.i = mul i64 %v69.i, -7723592293110705685, !dbg !255
  %v77.i = lshr i64 %v70.i, 40, !dbg !265
  %v78.i = uitofp nneg i64 %v77.i to float, !dbg !265
  %v79.i = fmul nnan contract float %v78.i, f0x34A00000, !dbg !265
  %v80.i = fadd contract float %v79.i, -2.500000e+00, !dbg !265
  store float %v33, ptr %v17, align 16, !dbg !266
  %v102.repack7.i = getelementptr inbounds nuw i8, ptr %v17, i64 4, !dbg !266
  store float 1.200000e+00, ptr %v102.repack7.i, align 4, !dbg !266
  store float 0.000000e+00, ptr %v19.elt10, align 8, !dbg !266
  %v102.repack11.i = getelementptr inbounds nuw i8, ptr %v17, i64 12, !dbg !266
  store float 0.000000e+00, ptr %v102.repack11.i, align 4, !dbg !266
  %v102.repack1.i = getelementptr inbounds nuw i8, ptr %v17, i64 16, !dbg !266
  store float 0.000000e+00, ptr %v102.repack1.i, align 16, !dbg !266
  %v102.repack1.repack13.i = getelementptr inbounds nuw i8, ptr %v17, i64 20, !dbg !266
  store float 0.000000e+00, ptr %v102.repack1.repack13.i, align 4, !dbg !266
  %v102.repack1.repack14.i = getelementptr inbounds nuw i8, ptr %v17, i64 24, !dbg !266
  store float -2.000000e+00, ptr %v102.repack1.repack14.i, align 8, !dbg !266
  %v102.repack1.repack15.i = getelementptr inbounds nuw i8, ptr %v17, i64 28, !dbg !266
  store float 0.000000e+00, ptr %v102.repack1.repack15.i, align 4, !dbg !266
  %v102.repack3.i = getelementptr inbounds nuw i8, ptr %v17, i64 32, !dbg !266
  store float 0.000000e+00, ptr %v102.repack3.i, align 16, !dbg !266
  %v102.repack3.repack16.i = getelementptr inbounds nuw i8, ptr %v17, i64 36, !dbg !266
  store float 0.000000e+00, ptr %v102.repack3.repack16.i, align 4, !dbg !266
  %v102.repack3.repack17.i = getelementptr inbounds nuw i8, ptr %v17, i64 40, !dbg !266
  store float 0.000000e+00, ptr %v102.repack3.repack17.i, align 8, !dbg !266
  %v102.repack3.repack18.i = getelementptr inbounds nuw i8, ptr %v17, i64 44, !dbg !266
  store float 1.000000e+00, ptr %v102.repack3.repack18.i, align 4, !dbg !266
  %v102.repack5.i = getelementptr inbounds nuw i8, ptr %v17, i64 48, !dbg !266
  store float 0.000000e+00, ptr %v102.repack5.i, align 16, !dbg !266
  %v102.repack5.repack19.i = getelementptr inbounds nuw i8, ptr %v17, i64 52, !dbg !266
  store float 0.000000e+00, ptr %v102.repack5.repack19.i, align 4, !dbg !266
  %v102.repack5.repack20.i = getelementptr inbounds nuw i8, ptr %v17, i64 56, !dbg !266
  store float 0.000000e+00, ptr %v102.repack5.repack20.i, align 8, !dbg !266
  %v102.repack5.repack21.i = getelementptr inbounds nuw i8, ptr %v17, i64 60, !dbg !266
  store float 0.000000e+00, ptr %v102.repack5.repack21.i, align 4, !dbg !266
  store float %v33, ptr %v28, align 4, !dbg !267
  store float 0.000000e+00, ptr %v29.elt15, align 4, !dbg !267
  store float %v80.i, ptr %v29.elt17, align 4, !dbg !267
  store i32 0, ptr %v29.elt19, align 4, !dbg !267
  store float 0.000000e+00, ptr %v29.elt21, align 4, !dbg !267
  %v110.i = fadd contract float %v33, %v80.i, !dbg !268
  %v111.i = fadd contract float %v110.i, -8.000000e-01, !dbg !268
  %v112.i = fadd contract float %v111.i, -4.000000e+00, !dbg !268
  %.idx.i = shl nuw nsw i64 %v22.i, 7, !dbg !269
  %4 = getelementptr i8, ptr %v0.fca.0.extract, i64 %.idx.i, !dbg !269
  %v125.i = getelementptr [64 x i8], ptr %4, i64 %v12, !dbg !269
  store float %v112.i, ptr %v125.i, align 16, !dbg !271
  %v125.repack36.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 4, !dbg !271
  store float 1.500000e+00, ptr %v125.repack36.i, align 4, !dbg !271
  %v125.repack38.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 8, !dbg !271
  store float -6.000000e+00, ptr %v125.repack38.i, align 8, !dbg !271
  %v125.repack40.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 12, !dbg !271
  store float 0.000000e+00, ptr %v125.repack40.i, align 4, !dbg !271
  %v125.repack30.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 16, !dbg !271
  store float 0.000000e+00, ptr %v125.repack30.i, align 16, !dbg !271
  %v125.repack30.repack42.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 20, !dbg !271
  store float 0.000000e+00, ptr %v125.repack30.repack42.i, align 4, !dbg !271
  %v125.repack30.repack43.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 24, !dbg !271
  store float 0.000000e+00, ptr %v125.repack30.repack43.i, align 8, !dbg !271
  %v125.repack30.repack44.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 28, !dbg !271
  store float 0.000000e+00, ptr %v125.repack30.repack44.i, align 4, !dbg !271
  %v125.repack32.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 32, !dbg !271
  store float 0.000000e+00, ptr %v125.repack32.i, align 16, !dbg !271
  %v125.repack32.repack45.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 36, !dbg !271
  store float 0.000000e+00, ptr %v125.repack32.repack45.i, align 4, !dbg !271
  %v125.repack32.repack46.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 40, !dbg !271
  store float 0.000000e+00, ptr %v125.repack32.repack46.i, align 8, !dbg !271
  %v125.repack32.repack47.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 44, !dbg !271
  store float 1.000000e+00, ptr %v125.repack32.repack47.i, align 4, !dbg !271
  %v125.repack34.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 48, !dbg !271
  store float 0.000000e+00, ptr %v125.repack34.i, align 16, !dbg !271
  %v125.repack34.repack48.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 52, !dbg !271
  store float 0.000000e+00, ptr %v125.repack34.repack48.i, align 4, !dbg !271
  %v125.repack34.repack49.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 56, !dbg !271
  store float 0.000000e+00, ptr %v125.repack34.repack49.i, align 8, !dbg !271
  %v125.repack34.repack50.i = getelementptr inbounds nuw i8, ptr %v125.i, i64 60, !dbg !271
  store float 0.000000e+00, ptr %v125.repack34.repack50.i, align 4, !dbg !271
  %v127.i = fadd contract float %v110.i, 8.000000e-01, !dbg !272
  %v128.i = fadd contract float %v127.i, 4.000000e+00, !dbg !273
  %v132.i = getelementptr i8, ptr %v125.i, i64 64, !dbg !269
  store float %v128.i, ptr %v132.i, align 16, !dbg !274
  %v132.repack57.i = getelementptr i8, ptr %v125.i, i64 68, !dbg !274
  store float 1.500000e+00, ptr %v132.repack57.i, align 4, !dbg !274
  %v132.repack59.i = getelementptr i8, ptr %v125.i, i64 72, !dbg !274
  store float -6.000000e+00, ptr %v132.repack59.i, align 8, !dbg !274
  %v132.repack61.i = getelementptr i8, ptr %v125.i, i64 76, !dbg !274
  store float 0.000000e+00, ptr %v132.repack61.i, align 4, !dbg !274
  %v132.repack51.i = getelementptr i8, ptr %v125.i, i64 80, !dbg !274
  store float 0.000000e+00, ptr %v132.repack51.i, align 16, !dbg !274
  %v132.repack51.repack63.i = getelementptr i8, ptr %v125.i, i64 84, !dbg !274
  store float 0.000000e+00, ptr %v132.repack51.repack63.i, align 4, !dbg !274
  %v132.repack51.repack64.i = getelementptr i8, ptr %v125.i, i64 88, !dbg !274
  store float 0.000000e+00, ptr %v132.repack51.repack64.i, align 8, !dbg !274
  %v132.repack51.repack65.i = getelementptr i8, ptr %v125.i, i64 92, !dbg !274
  store float 0.000000e+00, ptr %v132.repack51.repack65.i, align 4, !dbg !274
  %v132.repack53.i = getelementptr i8, ptr %v125.i, i64 96, !dbg !274
  store float 0.000000e+00, ptr %v132.repack53.i, align 16, !dbg !274
  %v132.repack53.repack66.i = getelementptr i8, ptr %v125.i, i64 100, !dbg !274
  store float 0.000000e+00, ptr %v132.repack53.repack66.i, align 4, !dbg !274
  %v132.repack53.repack67.i = getelementptr i8, ptr %v125.i, i64 104, !dbg !274
  store float 0.000000e+00, ptr %v132.repack53.repack67.i, align 8, !dbg !274
  %v132.repack53.repack68.i = getelementptr i8, ptr %v125.i, i64 108, !dbg !274
  store float 1.000000e+00, ptr %v132.repack53.repack68.i, align 4, !dbg !274
  %v132.repack55.i = getelementptr i8, ptr %v125.i, i64 112, !dbg !274
  store float 0.000000e+00, ptr %v132.repack55.i, align 16, !dbg !274
  %v132.repack55.repack69.i = getelementptr i8, ptr %v125.i, i64 116, !dbg !274
  store float 0.000000e+00, ptr %v132.repack55.repack69.i, align 4, !dbg !274
  %v132.repack55.repack70.i = getelementptr i8, ptr %v125.i, i64 120, !dbg !274
  store float 0.000000e+00, ptr %v132.repack55.repack70.i, align 8, !dbg !274
  %v132.repack55.repack71.i = getelementptr i8, ptr %v125.i, i64 124, !dbg !274
  store float 0.000000e+00, ptr %v132.repack55.repack71.i, align 4, !dbg !274
  %v66.i.pre = load float, ptr %v29.elt17, align 4, !dbg !275
  %v68.i44.pre = load i32, ptr %v29.elt19, align 4, !dbg !275
  %v70.i46.pre = load float, ptr %v29.elt21, align 4, !dbg !275
  br label %bb33, !dbg !277

bb33:                                             ; preds = %bb31, %bb8
  %v70.i46 = phi float [ %v70.i46.pre, %bb31 ], [ %v29.unpack22, %bb8 ], !dbg !275
  %v68.i44 = phi i32 [ %v68.i44.pre, %bb31 ], [ %v57, %bb8 ], !dbg !275
  %v66.i = phi float [ %v66.i.pre, %bb31 ], [ %v29.unpack18, %bb8 ], !dbg !275
  call void @llvm.lifetime.start.p0(ptr nonnull %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i), !dbg !278
  %v51.unpack.i = load float, ptr %v17, align 16, !dbg !280
  %v51.elt1.i = getelementptr inbounds nuw i8, ptr %v17, i64 4, !dbg !280
  %v51.unpack2.i = load float, ptr %v51.elt1.i, align 4, !dbg !280
  %v51.unpack4.i = load float, ptr %v19.elt10, align 8, !dbg !280
  %v52.i = getelementptr inbounds nuw i8, ptr %v17, i64 16, !dbg !280
  %v53.unpack.i = load float, ptr %v52.i, align 16, !dbg !280
  %v53.elt8.i = getelementptr inbounds nuw i8, ptr %v17, i64 20, !dbg !280
  %v53.unpack9.i = load float, ptr %v53.elt8.i, align 4, !dbg !280
  %v53.elt10.i = getelementptr inbounds nuw i8, ptr %v17, i64 24, !dbg !280
  %v53.unpack11.i = load float, ptr %v53.elt10.i, align 8, !dbg !280
  %v79.i49 = uitofp i32 %v68.i44 to float, !dbg !281
  %v80.i50 = fdiv contract float %v79.i49, 1.500000e+02, !dbg !281
  %v81.i = fsub contract float %v33, %v51.unpack.i, !dbg !282
  %v82.i = fsub contract float 1.200000e+00, %v51.unpack2.i, !dbg !283
  %v83.i = fsub contract float -1.200000e+01, %v51.unpack4.i, !dbg !284
  %v90.i = fsub contract float 1.000000e+00, %v80.i50, !dbg !285
  %v99.i = fadd contract float %v33, %v66.i, !dbg !286
  %v100.i = fsub contract float %v99.i, %v51.unpack.i, !dbg !286
  %v101.i = fsub contract float -6.000000e+00, %v51.unpack4.i, !dbg !287
  %.idx.i51 = shl nuw nsw i64 %v22.i, 5
  %5 = getelementptr i8, ptr %v0.fca.4.extract, i64 %.idx.i51
  store float %v81.i, ptr %5, align 4, !dbg !288
  %v116.1.i = getelementptr i8, ptr %5, i64 4, !dbg !278
  store float %v82.i, ptr %v116.1.i, align 4, !dbg !288
  %v116.2.i = getelementptr i8, ptr %5, i64 8, !dbg !278
  store float %v83.i, ptr %v116.2.i, align 4, !dbg !288
  %v116.3.i = getelementptr i8, ptr %5, i64 12, !dbg !278
  store float %v53.unpack.i, ptr %v116.3.i, align 4, !dbg !288
  %v116.4.i = getelementptr i8, ptr %5, i64 16, !dbg !278
  store float %v53.unpack9.i, ptr %v116.4.i, align 4, !dbg !288
  %v116.5.i = getelementptr i8, ptr %5, i64 20, !dbg !278
  store float %v53.unpack11.i, ptr %v116.5.i, align 4, !dbg !288
  %v116.6.i = getelementptr i8, ptr %5, i64 24, !dbg !278
  store float %v90.i, ptr %v116.6.i, align 4, !dbg !288
  %v116.7.i = getelementptr i8, ptr %5, i64 28, !dbg !278
  store float %v70.i46, ptr %v116.7.i, align 4, !dbg !288
  %.idx15.i = shl nuw nsw i64 %v22.i, 4
  %6 = getelementptr i8, ptr %v0.fca.8.extract, i64 %.idx15.i
  store float %v100.i, ptr %6, align 4, !dbg !289
  %v128.1.i = getelementptr i8, ptr %6, i64 4, !dbg !278
  store float %v101.i, ptr %v128.1.i, align 4, !dbg !289
  %v128.2.i = getelementptr i8, ptr %6, i64 8, !dbg !278
  store float 8.000000e-01, ptr %v128.2.i, align 4, !dbg !289
  %v128.3.i = getelementptr i8, ptr %6, i64 12, !dbg !278
  store float %v80.i50, ptr %v128.3.i, align 4, !dbg !289
  %v138.i = fneg float %v51.unpack.i, !dbg !290
  %v139.i = fneg float %v51.unpack2.i, !dbg !291
  %v140.i = fneg float %v51.unpack4.i, !dbg !292
  %v141.i = fneg float %v1.fca.3.extract, !dbg !293
  %v142.i = fmul contract float %v51.unpack.i, %v141.i, !dbg !293
  %v143.i = fneg float %v1.fca.4.extract, !dbg !294
  %v144.i = fmul contract float %v51.unpack2.i, %v143.i, !dbg !294
  %v145.i = fmul contract float %v1.fca.5.extract, %v51.unpack4.i, !dbg !295
  %v146.i = fsub contract float %v1.fca.6.extract, %v145.i, !dbg !296
  store float 1.000000e+00, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, align 4, !dbg !297
  %v182.fca.1.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 4, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.1.gep.i, align 4, !dbg !297
  %v182.fca.2.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 8, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.2.gep.i, align 4, !dbg !297
  %v182.fca.3.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 12, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.3.gep.i, align 4, !dbg !297
  %v182.fca.4.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 16, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.4.gep.i, align 4, !dbg !297
  %v182.fca.5.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 20, !dbg !297
  store float 1.000000e+00, ptr %v182.fca.5.gep.i, align 4, !dbg !297
  %v182.fca.6.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 24, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.6.gep.i, align 4, !dbg !297
  %v182.fca.7.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 28, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.7.gep.i, align 4, !dbg !297
  %v182.fca.8.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 32, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.8.gep.i, align 4, !dbg !297
  %v182.fca.9.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 36, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.9.gep.i, align 4, !dbg !297
  %v182.fca.10.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 40, !dbg !297
  store float 1.000000e+00, ptr %v182.fca.10.gep.i, align 4, !dbg !297
  %v182.fca.11.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 44, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.11.gep.i, align 4, !dbg !297
  %v182.fca.12.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 48, !dbg !297
  store float %v138.i, ptr %v182.fca.12.gep.i, align 4, !dbg !297
  %v182.fca.13.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 52, !dbg !297
  store float %v139.i, ptr %v182.fca.13.gep.i, align 4, !dbg !297
  %v182.fca.14.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 56, !dbg !297
  store float %v140.i, ptr %v182.fca.14.gep.i, align 4, !dbg !297
  %v182.fca.15.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 60, !dbg !297
  store float 1.000000e+00, ptr %v182.fca.15.gep.i, align 4, !dbg !297
  %v182.fca.16.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 64, !dbg !297
  store float %v1.fca.3.extract, ptr %v182.fca.16.gep.i, align 4, !dbg !297
  %v182.fca.17.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 68, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.17.gep.i, align 4, !dbg !297
  %v182.fca.18.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 72, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.18.gep.i, align 4, !dbg !297
  %v182.fca.19.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 76, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.19.gep.i, align 4, !dbg !297
  %v182.fca.20.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 80, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.20.gep.i, align 4, !dbg !297
  %v182.fca.21.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 84, !dbg !297
  store float %v1.fca.4.extract, ptr %v182.fca.21.gep.i, align 4, !dbg !297
  %v182.fca.22.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 88, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.22.gep.i, align 4, !dbg !297
  %v182.fca.23.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 92, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.23.gep.i, align 4, !dbg !297
  %v182.fca.24.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 96, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.24.gep.i, align 4, !dbg !297
  %v182.fca.25.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 100, !dbg !297
  store float 0.000000e+00, ptr %v182.fca.25.gep.i, align 4, !dbg !297
  %v182.fca.26.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 104, !dbg !297
  store float %v1.fca.5.extract, ptr %v182.fca.26.gep.i, align 4, !dbg !297
  %v182.fca.27.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 108, !dbg !297
  store float -1.000000e+00, ptr %v182.fca.27.gep.i, align 4, !dbg !297
  %v182.fca.28.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 112, !dbg !297
  store float %v142.i, ptr %v182.fca.28.gep.i, align 4, !dbg !297
  %v182.fca.29.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 116, !dbg !297
  store float %v144.i, ptr %v182.fca.29.gep.i, align 4, !dbg !297
  %v182.fca.30.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 120, !dbg !297
  store float %v146.i, ptr %v182.fca.30.gep.i, align 4, !dbg !297
  %v182.fca.31.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 124, !dbg !297
  store float %v51.unpack4.i, ptr %v182.fca.31.gep.i, align 4, !dbg !297
  %v182.fca.32.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 128, !dbg !297
  store float %v51.unpack.i, ptr %v182.fca.32.gep.i, align 4, !dbg !297
  %v182.fca.33.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 132, !dbg !297
  store float %v51.unpack2.i, ptr %v182.fca.33.gep.i, align 4, !dbg !297
  %v182.fca.34.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 136, !dbg !297
  store float %v51.unpack4.i, ptr %v182.fca.34.gep.i, align 4, !dbg !297
  %v182.fca.35.gep.i = getelementptr inbounds nuw i8, ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 140, !dbg !297
  store float 1.000000e+00, ptr %v182.fca.35.gep.i, align 4, !dbg !297
  %.idx16.i = mul nuw nsw i64 %v22.i, 144
  %7 = getelementptr i8, ptr %v0.fca.3.extract, i64 %.idx16.i
  br label %bb14.i, !dbg !298

bb14.i:                                           ; preds = %bb14.i, %bb33
  %v18319.i = phi i64 [ 0, %bb33 ], [ %v194.i.3, %bb14.i ]
  %v187.i = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v18319.i, !dbg !299
  %v188.i = load float, ptr %v187.i, align 4, !dbg !299
  %v193.i = getelementptr [4 x i8], ptr %7, i64 %v18319.i, !dbg !278
  store float %v188.i, ptr %v193.i, align 4, !dbg !300
  %v194.i = or disjoint i64 %v18319.i, 1, !dbg !301
  %v187.i.1 = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v194.i, !dbg !299
  %v188.i.1 = load float, ptr %v187.i.1, align 4, !dbg !299
  %v193.i.1 = getelementptr [4 x i8], ptr %7, i64 %v194.i, !dbg !278
  store float %v188.i.1, ptr %v193.i.1, align 4, !dbg !300
  %v194.i.1 = or disjoint i64 %v18319.i, 2, !dbg !301
  %v187.i.2 = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v194.i.1, !dbg !299
  %v188.i.2 = load float, ptr %v187.i.2, align 4, !dbg !299
  %v193.i.2 = getelementptr [4 x i8], ptr %7, i64 %v194.i.1, !dbg !278
  store float %v188.i.2, ptr %v193.i.2, align 4, !dbg !300
  %v194.i.2 = or disjoint i64 %v18319.i, 3, !dbg !301
  %v187.i.3 = getelementptr inbounds nuw [4 x i8], ptr %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i, i64 %v194.i.2, !dbg !299
  %v188.i.3 = load float, ptr %v187.i.3, align 4, !dbg !299
  %v193.i.3 = getelementptr [4 x i8], ptr %7, i64 %v194.i.2, !dbg !278
  store float %v188.i.3, ptr %v193.i.3, align 4, !dbg !300
  %v194.i.3 = add nuw nsw i64 %v18319.i, 4, !dbg !301
  %exitcond.i.3 = icmp eq i64 %v194.i.3, 36, !dbg !298
  br i1 %exitcond.i.3, label %sim_env__kernels__device__observe_one.exit, label %bb14.i, !dbg !298

sim_env__kernels__device__observe_one.exit:       ; preds = %bb14.i
  call void @llvm.lifetime.end.p0(ptr nonnull %__cuda_oxide_local_x3532093134340963616d657261095b6633323b2033365d_.i), !dbg !302
  br label %bb35, !dbg !303

bb35:                                             ; preds = %entry, %sim_env__kernels__device__observe_one.exit
  ret void, !dbg !303
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.y() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65) i32 @llvm.nvvm.read.ptx.sreg.ntid.z() #2

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 65536) i32 @llvm.nvvm.read.ptx.sreg.nctaid.z() #2

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #3

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(ptr captures(none)) #4

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(ptr captures(none)) #4

attributes #0 = { convergent mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none, target_mem: none) "target-cpu"="sm_120" }
attributes #1 = { convergent nofree norecurse nosync nounwind memory(readwrite, inaccessiblemem: none, target_mem: none) "target-cpu"="sm_120" }
attributes #2 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) "target-cpu"="sm_120" }
attributes #3 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #5 = { convergent }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_Rust, file: !1, producer: "cuda-oxide", isOptimized: true, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "kernels.rs", directory: "/home/jaren/visualencoder/sim/crates/sim-env/src")
!2 = !{i32 2, !"Dwarf Version", i32 2}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "control", linkageName: "control", scope: !1, file: !1, line: 131, type: !5, scopeLine: 131, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{}
!8 = !DILocation(line: 652, column: 13, scope: !9, inlinedAt: !11)
!9 = distinct !DISubprogram(name: "cuda_device::__internal::index_1d::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal8index_1dNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env", scope: !10, file: !10, line: 648, type: !5, scopeLine: 648, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!10 = !DIFile(filename: "thread.rs", directory: "/home/jaren/.cargo/git/checkouts/cuda-rust-27a4e94e7e45d2f6/348895e/cuda-oxide/crates/cuda-device/src")
!11 = distinct !DILocation(line: 132, column: 17, scope: !12)
!12 = !DILexicalBlock(scope: !4, file: !1, line: 131, column: 82)
!13 = !DILocation(line: 653, column: 13, scope: !9, inlinedAt: !11)
!14 = !DILocation(line: 654, column: 13, scope: !9, inlinedAt: !11)
!15 = !DILocation(line: 602, column: 21, scope: !16, inlinedAt: !17)
!16 = distinct !DISubprogram(name: "cuda_device::__internal::checked_axis", scope: !10, file: !10, line: 601, type: !5, scopeLine: 601, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!17 = distinct !DILocation(line: 651, column: 19, scope: !9, inlinedAt: !11)
!18 = !DILocation(line: 603, column: 26, scope: !16, inlinedAt: !17)
!19 = !DILocation(line: 608, column: 13, scope: !16, inlinedAt: !17)
!20 = !DILocation(line: 604, column: 22, scope: !16, inlinedAt: !17)
!21 = !DILocation(line: 619, column: 13, scope: !22, inlinedAt: !23)
!22 = distinct !DISubprogram(name: "cuda_device::__internal::one_dimensional_launch::<cuda_device::__internal::UnknownDomain, cuda_device::__internal::NativeCoordinates>", linkageName: "_RINvNtNtCsck0txXvZ9fJ_11cuda_device6thread10___internal22one_dimensional_launchNtB2_13UnknownDomainNtB2_17NativeCoordinatesECsj02jzEoshPt_7sim_env", scope: !10, file: !10, line: 613, type: !5, scopeLine: 613, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
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
!33 = !DILocation(line: 133, column: 17, scope: !34)
!34 = !DILexicalBlock(scope: !12, file: !1, line: 132, column: 9)
!35 = !DILocation(line: 133, column: 12, scope: !34)
!36 = !DILocation(line: 970, column: 18, scope: !37, inlinedAt: !39)
!37 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut sim_state::BodyAction>::add", scope: !38, file: !38, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!38 = !DIFile(filename: "mut_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!39 = !DILocation(line: 143, column: 29, scope: !40)
!40 = !DILexicalBlock(scope: !41, file: !1, line: 137, column: 9)
!41 = !DILexicalBlock(scope: !42, file: !1, line: 136, column: 9)
!42 = !DILexicalBlock(scope: !43, file: !1, line: 135, column: 9)
!43 = !DILexicalBlock(scope: !34, file: !1, line: 134, column: 9)
!44 = !DILocation(line: 134, column: 29, scope: !34)
!45 = !DILocation(line: 871, column: 18, scope: !46, inlinedAt: !48)
!46 = distinct !DISubprogram(name: "std::ptr::const_ptr::<impl *const i64>::add", scope: !47, file: !47, line: 837, type: !5, scopeLine: 837, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!47 = !DIFile(filename: "const_ptr.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/ptr")
!48 = !DILocation(line: 135, column: 40, scope: !43)
!49 = !DILocation(line: 135, column: 31, scope: !43)
!50 = !DILocation(line: 136, column: 23, scope: !42)
!51 = !DILocation(line: 136, column: 22, scope: !42)
!52 = !DILocation(line: 140, column: 9, scope: !40)
!53 = !DILocation(line: 142, column: 13, scope: !40)
!54 = !DILocation(line: 144, column: 31, scope: !40)
!55 = !DILocation(line: 144, column: 25, scope: !40)
!56 = !DILocation(line: 145, column: 35, scope: !40)
!57 = !DILocation(line: 145, column: 28, scope: !40)
!58 = !DILocation(line: 145, column: 21, scope: !40)
!59 = !DILocation(line: 145, column: 62, scope: !40)
!60 = !DILocation(line: 146, column: 27, scope: !40)
!61 = !DILocation(line: 146, column: 21, scope: !40)
!62 = !DILocation(line: 143, column: 13, scope: !40)
!63 = !DILocation(line: 150, column: 6, scope: !4)
!64 = distinct !DISubprogram(name: "reset", linkageName: "reset", scope: !1, file: !1, line: 115, type: !5, scopeLine: 115, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!65 = !DILocation(line: 652, column: 13, scope: !9, inlinedAt: !66)
!66 = distinct !DILocation(line: 116, column: 17, scope: !67)
!67 = !DILexicalBlock(scope: !64, file: !1, line: 115, column: 59)
!68 = !DILocation(line: 653, column: 13, scope: !9, inlinedAt: !66)
!69 = !DILocation(line: 654, column: 13, scope: !9, inlinedAt: !66)
!70 = !DILocation(line: 602, column: 21, scope: !16, inlinedAt: !71)
!71 = distinct !DILocation(line: 651, column: 19, scope: !9, inlinedAt: !66)
!72 = !DILocation(line: 603, column: 26, scope: !16, inlinedAt: !71)
!73 = !DILocation(line: 608, column: 13, scope: !16, inlinedAt: !71)
!74 = !DILocation(line: 604, column: 22, scope: !16, inlinedAt: !71)
!75 = !DILocation(line: 619, column: 13, scope: !22, inlinedAt: !76)
!76 = distinct !DILocation(line: 656, column: 21, scope: !24, inlinedAt: !66)
!77 = !DILocation(line: 624, column: 13, scope: !26, inlinedAt: !76)
!78 = !DILocation(line: 626, column: 9, scope: !28, inlinedAt: !76)
!79 = !DILocation(line: 312, column: 21, scope: !30, inlinedAt: !80)
!80 = distinct !DILocation(line: 657, column: 18, scope: !32, inlinedAt: !66)
!81 = !DILocation(line: 117, column: 17, scope: !82)
!82 = !DILexicalBlock(scope: !67, file: !1, line: 116, column: 9)
!83 = !DILocation(line: 117, column: 12, scope: !82)
!84 = !DILocation(line: 970, column: 18, scope: !85, inlinedAt: !86)
!85 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut f32>::add", scope: !38, file: !38, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!86 = !DILocation(line: 124, column: 32, scope: !82)
!87 = !DILocation(line: 119, column: 13, scope: !82)
!88 = !DILocation(line: 120, column: 13, scope: !82)
!89 = !DILocation(line: 121, column: 13, scope: !82)
!90 = !DILocation(line: 122, column: 13, scope: !82)
!91 = !DILocation(line: 123, column: 13, scope: !82)
!92 = !DILocation(line: 124, column: 13, scope: !82)
!93 = !DILocation(line: 52, column: 20, scope: !94, inlinedAt: !95)
!94 = distinct !DISubprogram(name: "sim_env::kernels::device::reset_one", linkageName: "sim_env__kernels__device__reset_one", scope: !1, file: !1, line: 51, type: !5, scopeLine: 51, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!95 = distinct !DILocation(line: 125, column: 13, scope: !82)
!96 = !DILocation(line: 2728, column: 13, scope: !97, inlinedAt: !101)
!97 = !DILexicalBlockFile(scope: !99, file: !98, discriminator: 0)
!98 = !DIFile(filename: "uint_macros.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/num")
!99 = distinct !DISubprogram(name: "core::num::<impl u64>::wrapping_mul", scope: !100, file: !100, line: 1322, type: !5, scopeLine: 1322, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!100 = !DIFile(filename: "mod.rs", directory: "/nix/store/s9f8yvw526vvjh007ryw66wbm0dpn8gl-rust-default-1.99.1/lib/rustlib/src/rust/library/core/src/num")
!101 = distinct !DILocation(line: 13, column: 25, scope: !102, inlinedAt: !104)
!102 = distinct !DISubprogram(name: "task::episode_gap", scope: !103, file: !103, line: 8, type: !5, scopeLine: 8, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!103 = !DIFile(filename: "task.rs", directory: "/home/jaren/visualencoder/sim/crates/sim-env/src")
!104 = distinct !DILocation(line: 53, column: 19, scope: !105, inlinedAt: !95)
!105 = !DILexicalBlock(scope: !94, file: !1, line: 52, column: 9)
!106 = !DILocation(line: 9, column: 17, scope: !102, inlinedAt: !104)
!107 = !DILocation(line: 11, column: 11, scope: !102, inlinedAt: !104)
!108 = !DILocation(line: 12, column: 14, scope: !102, inlinedAt: !104)
!109 = !DILocation(line: 12, column: 9, scope: !102, inlinedAt: !104)
!110 = !DILocation(line: 13, column: 14, scope: !102, inlinedAt: !104)
!111 = !DILocation(line: 13, column: 9, scope: !102, inlinedAt: !104)
!112 = !DILocation(line: 15, column: 5, scope: !102, inlinedAt: !104)
!113 = !DILocation(line: 970, column: 18, scope: !114, inlinedAt: !115)
!114 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut sim_state::BodyState>::add", scope: !38, file: !38, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!115 = distinct !DILocation(line: 71, column: 29, scope: !116, inlinedAt: !95)
!116 = !DILexicalBlock(scope: !117, file: !1, line: 69, column: 13)
!117 = !DILexicalBlock(scope: !118, file: !1, line: 64, column: 13)
!118 = !DILexicalBlock(scope: !105, file: !1, line: 53, column: 9)
!119 = !DILocation(line: 55, column: 13, scope: !118, inlinedAt: !95)
!120 = !DILocation(line: 61, column: 13, scope: !118, inlinedAt: !95)
!121 = !DILocation(line: 65, column: 28, scope: !118, inlinedAt: !95)
!122 = !DILocation(line: 68, column: 13, scope: !117, inlinedAt: !95)
!123 = !DILocation(line: 70, column: 33, scope: !116, inlinedAt: !95)
!124 = !DILocation(line: 70, column: 13, scope: !116, inlinedAt: !95)
!125 = !DILocation(line: 71, column: 13, scope: !116, inlinedAt: !95)
!126 = !DILocation(line: 970, column: 18, scope: !85, inlinedAt: !127)
!127 = distinct !DILocation(line: 109, column: 39, scope: !128, inlinedAt: !142)
!128 = !DILexicalBlock(scope: !129, file: !1, line: 100, column: 9)
!129 = !DILexicalBlock(scope: !130, file: !1, line: 99, column: 9)
!130 = !DILexicalBlock(scope: !131, file: !1, line: 98, column: 9)
!131 = !DILexicalBlock(scope: !132, file: !1, line: 97, column: 9)
!132 = !DILexicalBlock(scope: !133, file: !1, line: 96, column: 9)
!133 = !DILexicalBlock(scope: !134, file: !1, line: 84, column: 9)
!134 = !DILexicalBlock(scope: !135, file: !1, line: 83, column: 9)
!135 = !DILexicalBlock(scope: !136, file: !1, line: 81, column: 9)
!136 = !DILexicalBlock(scope: !137, file: !1, line: 80, column: 9)
!137 = !DILexicalBlock(scope: !138, file: !1, line: 79, column: 9)
!138 = !DILexicalBlock(scope: !139, file: !1, line: 78, column: 9)
!139 = !DILexicalBlock(scope: !140, file: !1, line: 77, column: 9)
!140 = !DILexicalBlock(scope: !141, file: !1, line: 76, column: 9)
!141 = distinct !DISubprogram(name: "sim_env::kernels::device::observe_one", linkageName: "sim_env__kernels__device__observe_one", scope: !1, file: !1, line: 75, type: !5, scopeLine: 75, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!142 = distinct !DILocation(line: 126, column: 13, scope: !82)
!143 = !DILocation(line: 76, column: 29, scope: !141, inlinedAt: !142)
!144 = !DILocation(line: 77, column: 29, scope: !140, inlinedAt: !142)
!145 = !DILocation(line: 80, column: 21, scope: !137, inlinedAt: !142)
!146 = !DILocation(line: 81, column: 24, scope: !136, inlinedAt: !142)
!147 = !DILocation(line: 81, column: 34, scope: !136, inlinedAt: !142)
!148 = !DILocation(line: 81, column: 43, scope: !136, inlinedAt: !142)
!149 = !DILocation(line: 82, column: 67, scope: !136, inlinedAt: !142)
!150 = !DILocation(line: 83, column: 27, scope: !135, inlinedAt: !142)
!151 = !DILocation(line: 83, column: 48, scope: !135, inlinedAt: !142)
!152 = !DILocation(line: 86, column: 22, scope: !133, inlinedAt: !142)
!153 = !DILocation(line: 91, column: 22, scope: !133, inlinedAt: !142)
!154 = !DILocation(line: 102, column: 33, scope: !129, inlinedAt: !142)
!155 = !DILocation(line: 102, column: 37, scope: !129, inlinedAt: !142)
!156 = !DILocation(line: 102, column: 41, scope: !129, inlinedAt: !142)
!157 = !DILocation(line: 104, column: 32, scope: !129, inlinedAt: !142)
!158 = !DILocation(line: 104, column: 40, scope: !129, inlinedAt: !142)
!159 = !DILocation(line: 104, column: 52, scope: !129, inlinedAt: !142)
!160 = !DILocation(line: 104, column: 48, scope: !129, inlinedAt: !142)
!161 = !DILocation(line: 100, column: 22, scope: !129, inlinedAt: !142)
!162 = !DILocation(line: 108, column: 15, scope: !128, inlinedAt: !142)
!163 = !DILocation(line: 109, column: 57, scope: !128, inlinedAt: !142)
!164 = !DILocation(line: 109, column: 22, scope: !128, inlinedAt: !142)
!165 = !DILocation(line: 110, column: 13, scope: !128, inlinedAt: !142)
!166 = !DILocation(line: 112, column: 6, scope: !141, inlinedAt: !142)
!167 = !DILocation(line: 128, column: 6, scope: !64)
!168 = distinct !DISubprogram(name: "finish", linkageName: "finish", scope: !1, file: !1, line: 153, type: !5, scopeLine: 153, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!169 = !DILocation(line: 652, column: 13, scope: !9, inlinedAt: !170)
!170 = distinct !DILocation(line: 154, column: 17, scope: !171)
!171 = !DILexicalBlock(scope: !168, file: !1, line: 153, column: 60)
!172 = !DILocation(line: 653, column: 13, scope: !9, inlinedAt: !170)
!173 = !DILocation(line: 654, column: 13, scope: !9, inlinedAt: !170)
!174 = !DILocation(line: 602, column: 21, scope: !16, inlinedAt: !175)
!175 = distinct !DILocation(line: 651, column: 19, scope: !9, inlinedAt: !170)
!176 = !DILocation(line: 603, column: 26, scope: !16, inlinedAt: !175)
!177 = !DILocation(line: 608, column: 13, scope: !16, inlinedAt: !175)
!178 = !DILocation(line: 604, column: 22, scope: !16, inlinedAt: !175)
!179 = !DILocation(line: 619, column: 13, scope: !22, inlinedAt: !180)
!180 = distinct !DILocation(line: 656, column: 21, scope: !24, inlinedAt: !170)
!181 = !DILocation(line: 624, column: 13, scope: !26, inlinedAt: !180)
!182 = !DILocation(line: 626, column: 9, scope: !28, inlinedAt: !180)
!183 = !DILocation(line: 312, column: 21, scope: !30, inlinedAt: !184)
!184 = distinct !DILocation(line: 657, column: 18, scope: !32, inlinedAt: !170)
!185 = !DILocation(line: 155, column: 17, scope: !186)
!186 = !DILexicalBlock(scope: !171, file: !1, line: 154, column: 9)
!187 = !DILocation(line: 155, column: 12, scope: !186)
!188 = !DILocation(line: 970, column: 18, scope: !189, inlinedAt: !190)
!189 = distinct !DISubprogram(name: "std::ptr::mut_ptr::<impl *mut u64>::add", scope: !38, file: !38, line: 936, type: !5, scopeLine: 936, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!190 = !DILocation(line: 174, column: 50, scope: !191)
!191 = !DILexicalBlock(scope: !192, file: !1, line: 165, column: 9)
!192 = !DILexicalBlock(scope: !193, file: !1, line: 164, column: 9)
!193 = !DILexicalBlock(scope: !194, file: !1, line: 162, column: 9)
!194 = !DILexicalBlock(scope: !195, file: !1, line: 159, column: 9)
!195 = !DILexicalBlock(scope: !196, file: !1, line: 158, column: 9)
!196 = !DILexicalBlock(scope: !197, file: !1, line: 157, column: 9)
!197 = !DILexicalBlock(scope: !186, file: !1, line: 156, column: 9)
!198 = !DILocation(line: 156, column: 29, scope: !186)
!199 = !DILocation(line: 157, column: 33, scope: !197)
!200 = !DILocation(line: 158, column: 42, scope: !196)
!201 = !DILocation(line: 158, column: 23, scope: !196)
!202 = !DILocation(line: 159, column: 25, scope: !195)
!203 = !DILocation(line: 160, column: 26, scope: !195)
!204 = !DILocation(line: 22, column: 9, scope: !205, inlinedAt: !208)
!205 = !DILexicalBlock(scope: !206, file: !103, line: 21, column: 5)
!206 = !DILexicalBlock(scope: !207, file: !103, line: 20, column: 5)
!207 = distinct !DISubprogram(name: "sim_env::task::hits_wall", linkageName: "sim_env__task__hits_wall", scope: !103, file: !103, line: 19, type: !5, scopeLine: 19, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!208 = distinct !DILocation(line: 160, column: 16, scope: !195)
!209 = !DILocation(line: 22, column: 41, scope: !205, inlinedAt: !208)
!210 = !DILocation(line: 25, column: 14, scope: !205, inlinedAt: !208)
!211 = !DILocation(line: 27, column: 8, scope: !212, inlinedAt: !208)
!212 = !DILexicalBlock(scope: !213, file: !103, line: 26, column: 5)
!213 = !DILexicalBlock(scope: !205, file: !103, line: 25, column: 5)
!214 = !DILocation(line: 28, column: 18, scope: !212, inlinedAt: !208)
!215 = !DILocation(line: 29, column: 18, scope: !216, inlinedAt: !208)
!216 = !DILexicalBlock(scope: !212, file: !103, line: 28, column: 9)
!217 = !DILocation(line: 30, column: 27, scope: !218, inlinedAt: !208)
!218 = !DILexicalBlock(scope: !216, file: !103, line: 29, column: 9)
!219 = !DILocation(line: 30, column: 24, scope: !218, inlinedAt: !208)
!220 = !DILocation(line: 31, column: 12, scope: !221, inlinedAt: !208)
!221 = !DILexicalBlock(scope: !218, file: !103, line: 30, column: 9)
!222 = !DILocation(line: 32, column: 12, scope: !221, inlinedAt: !208)
!223 = !DILocation(line: 27, column: 5, scope: !212, inlinedAt: !208)
!224 = !DILocation(line: 34, column: 8, scope: !212, inlinedAt: !208)
!225 = !DILocation(line: 35, column: 22, scope: !212, inlinedAt: !208)
!226 = !DILocation(line: 35, column: 14, scope: !212, inlinedAt: !208)
!227 = !DILocation(line: 36, column: 22, scope: !228, inlinedAt: !208)
!228 = !DILexicalBlock(scope: !212, file: !103, line: 35, column: 5)
!229 = !DILocation(line: 36, column: 14, scope: !228, inlinedAt: !208)
!230 = !DILocation(line: 38, column: 5, scope: !231, inlinedAt: !208)
!231 = !DILexicalBlock(scope: !232, file: !103, line: 37, column: 5)
!232 = !DILexicalBlock(scope: !228, file: !103, line: 36, column: 5)
!233 = !DILocation(line: 162, column: 24, scope: !194)
!234 = !DILocation(line: 39, column: 2, scope: !207, inlinedAt: !208)
!235 = !DILocation(line: 163, column: 9, scope: !193)
!236 = !DILocation(line: 164, column: 20, scope: !193)
!237 = !DILocation(line: 165, column: 22, scope: !192)
!238 = !DILocation(line: 166, column: 18, scope: !192)
!239 = !DILocation(line: 168, column: 13, scope: !191)
!240 = !DILocation(line: 169, column: 39, scope: !191)
!241 = !DILocation(line: 169, column: 13, scope: !191)
!242 = !DILocation(line: 170, column: 42, scope: !191)
!243 = !DILocation(line: 170, column: 13, scope: !191)
!244 = !DILocation(line: 171, column: 44, scope: !191)
!245 = !DILocation(line: 171, column: 13, scope: !191)
!246 = !DILocation(line: 172, column: 13, scope: !191)
!247 = !DILocation(line: 173, column: 16, scope: !191)
!248 = !DILocation(line: 174, column: 31, scope: !191)
!249 = !DILocation(line: 2652, column: 13, scope: !250, inlinedAt: !252)
!250 = !DILexicalBlockFile(scope: !251, file: !98, discriminator: 0)
!251 = distinct !DISubprogram(name: "core::num::<impl u64>::wrapping_add", scope: !100, file: !100, line: 1322, type: !5, scopeLine: 1322, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !7)
!252 = !DILocation(line: 174, column: 58, scope: !191)
!253 = !DILocation(line: 175, column: 17, scope: !254)
!254 = !DILexicalBlock(scope: !191, file: !1, line: 174, column: 17)
!255 = !DILocation(line: 2728, column: 13, scope: !97, inlinedAt: !256)
!256 = distinct !DILocation(line: 13, column: 25, scope: !102, inlinedAt: !257)
!257 = distinct !DILocation(line: 53, column: 19, scope: !105, inlinedAt: !258)
!258 = distinct !DILocation(line: 176, column: 17, scope: !254)
!259 = !DILocation(line: 9, column: 17, scope: !102, inlinedAt: !257)
!260 = !DILocation(line: 11, column: 11, scope: !102, inlinedAt: !257)
!261 = !DILocation(line: 12, column: 14, scope: !102, inlinedAt: !257)
!262 = !DILocation(line: 12, column: 9, scope: !102, inlinedAt: !257)
!263 = !DILocation(line: 13, column: 14, scope: !102, inlinedAt: !257)
!264 = !DILocation(line: 13, column: 9, scope: !102, inlinedAt: !257)
!265 = !DILocation(line: 15, column: 5, scope: !102, inlinedAt: !257)
!266 = !DILocation(line: 55, column: 13, scope: !118, inlinedAt: !258)
!267 = !DILocation(line: 61, column: 13, scope: !118, inlinedAt: !258)
!268 = !DILocation(line: 65, column: 28, scope: !118, inlinedAt: !258)
!269 = !DILocation(line: 970, column: 18, scope: !114, inlinedAt: !270)
!270 = distinct !DILocation(line: 71, column: 29, scope: !116, inlinedAt: !258)
!271 = !DILocation(line: 68, column: 13, scope: !117, inlinedAt: !258)
!272 = !DILocation(line: 70, column: 33, scope: !116, inlinedAt: !258)
!273 = !DILocation(line: 70, column: 13, scope: !116, inlinedAt: !258)
!274 = !DILocation(line: 71, column: 13, scope: !116, inlinedAt: !258)
!275 = !DILocation(line: 77, column: 29, scope: !140, inlinedAt: !276)
!276 = distinct !DILocation(line: 180, column: 13, scope: !191)
!277 = !DILocation(line: 173, column: 13, scope: !191)
!278 = !DILocation(line: 970, column: 18, scope: !85, inlinedAt: !279)
!279 = distinct !DILocation(line: 109, column: 39, scope: !128, inlinedAt: !276)
!280 = !DILocation(line: 76, column: 29, scope: !141, inlinedAt: !276)
!281 = !DILocation(line: 80, column: 21, scope: !137, inlinedAt: !276)
!282 = !DILocation(line: 81, column: 24, scope: !136, inlinedAt: !276)
!283 = !DILocation(line: 81, column: 34, scope: !136, inlinedAt: !276)
!284 = !DILocation(line: 81, column: 43, scope: !136, inlinedAt: !276)
!285 = !DILocation(line: 82, column: 67, scope: !136, inlinedAt: !276)
!286 = !DILocation(line: 83, column: 27, scope: !135, inlinedAt: !276)
!287 = !DILocation(line: 83, column: 48, scope: !135, inlinedAt: !276)
!288 = !DILocation(line: 86, column: 22, scope: !133, inlinedAt: !276)
!289 = !DILocation(line: 91, column: 22, scope: !133, inlinedAt: !276)
!290 = !DILocation(line: 102, column: 33, scope: !129, inlinedAt: !276)
!291 = !DILocation(line: 102, column: 37, scope: !129, inlinedAt: !276)
!292 = !DILocation(line: 102, column: 41, scope: !129, inlinedAt: !276)
!293 = !DILocation(line: 104, column: 32, scope: !129, inlinedAt: !276)
!294 = !DILocation(line: 104, column: 40, scope: !129, inlinedAt: !276)
!295 = !DILocation(line: 104, column: 52, scope: !129, inlinedAt: !276)
!296 = !DILocation(line: 104, column: 48, scope: !129, inlinedAt: !276)
!297 = !DILocation(line: 100, column: 22, scope: !129, inlinedAt: !276)
!298 = !DILocation(line: 108, column: 15, scope: !128, inlinedAt: !276)
!299 = !DILocation(line: 109, column: 57, scope: !128, inlinedAt: !276)
!300 = !DILocation(line: 109, column: 22, scope: !128, inlinedAt: !276)
!301 = !DILocation(line: 110, column: 13, scope: !128, inlinedAt: !276)
!302 = !DILocation(line: 112, column: 6, scope: !141, inlinedAt: !276)
!303 = !DILocation(line: 182, column: 6, scope: !168)
