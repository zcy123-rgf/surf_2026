# 旧目录整理与清理指南

不要先删除 `F:\2026_surf`。先运行 `scripts/create_clean_workstation_copy.ps1` 创建新子目录，并分别完成环境检查、RANSAC参考运行和曲线模型运行。

## 可以归档而不是立即删除

以下通常是试错结果或汇报材料，不是运行源码：

- `workstation_outputs` 中旧的时间戳结果；
- `generated_runs`、`denoise_experiments`；
- `meeting_materials_*`、PPT生成物；
- `official_kitti_pose_audit`；
- 根目录临时patch和手工备份。

先移动到单独归档盘，并保留至少一周；确认精简版结果和SHA/JSON审计无误后再删除。不要删除：

- 工作中的 `CLRNet` 目录及 `weights`；
- `surf2026-win` Conda环境；
- KITTI数据集；
- 最新成功输出以及它的 `STATUS.json`、`RUN_STATUS.json`；
- 当前Git目录，直到精简分支已经成功拉取并验证。

## 建议的安全移动方式

```powershell
$Archive = "F:\2026_surf_archive_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory -Force $Archive

# 先逐项确认存在，再移动；不要直接整目录删除。
Get-ChildItem F:\2026_surf\workstation_outputs -Directory |
  Sort-Object LastWriteTime |
  Select-Object Name,LastWriteTime
```

本指南故意不提供递归删除命令，避免误删唯一可复现结果。
