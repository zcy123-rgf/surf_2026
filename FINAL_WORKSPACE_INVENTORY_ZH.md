# F:\surf_final 最终目录规范

## 允许保留的顶层内容

- `annotations/`：项目使用的人工伪标注。
- `CLRNet/`：单独保存的CLRNet运行时、Windows兼容层和`weights/culane_r18.pth`。
- `scripts/`：最终流程、RANSAC基线、全量运行、固定评测和工作区审计脚本。
- `surf_bev/`：公共检测、IPM、pose变换和RANSAC模块。
- `results/`：人工确认后固定保存的代表性结果，可为空。
- `workstation_outputs/`：带时间戳的正式运行结果和审计报告。
- `backups/`：审计脚本生成的源码与CLRNet运行时备份。
- `requirements-windows.txt`、`README_SURF_FINAL_ZH.md`、`SURF_FINAL_PIPELINE_ZH.md`、`SURF_FINAL_DECISIONS_ZH.md`、`FIXED_EVALUATION_PROTOCOL_ZH.md`、`SOURCE_MANIFEST.json`。

## 正式输出前缀

- `final_seq01_851_1005_*`：155帧正式路段。
- `final_seq01_0_1100_*`：完整Sequence 01。
- `surf_final_all_sequences_*`：Sequence 00–10全量运行。
- `fixed_project_metrics_*`：固定内部一致性评测；最终协议必须为`2.0-domain-aligned`。
- `pose_curvature_all_sequences_*`：全Sequence pose曲率扫描。
- `workspace_audit_*`：只读工作区审计。

其他名称的输出不会被脚本自动删除，只会列入`output_inventory.csv`等待人工确认。

## 审计和备份

在Anaconda PowerShell Prompt中运行：

```powershell
Set-Location F:\surf_final
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

& .\scripts\audit_surf_final_workspace.ps1 `
  -ProjectRoot "F:\surf_final" `
  -CreateSourceBackup
```

脚本不会删除或移动任何原有文件。它检查源码、CLRNet权重、00–10全量结果和固定评测，并把报告写入新的`workstation_outputs\workspace_audit_时间戳`。

源码/运行时备份包含CLRNet及其权重，但不包含体积较大的`workstation_outputs`。完整实验输出仍保留在`F:\surf_final\workstation_outputs`，代码和数据齐全时也可以重新生成。

只有当`WORKSPACE_AUDIT.json`显示`ready_for_final_retention`时，才能认为`F:\surf_final`符合最终目录规范。若显示`review_required`，先查看CSV清单，不要直接删除被标记的内容。
