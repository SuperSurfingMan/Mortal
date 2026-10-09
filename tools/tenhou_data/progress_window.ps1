# Mortal 天凤采集进度看板（每 30 秒刷新；关闭窗口即退出）
chcp 65001 > $null
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$csv    = 'C:\Maj\Mortal\dataset\index\houou_games.csv'
$logdir = 'C:\Maj\Mortal\tools\tenhou_data\logs'
$stdout = Join-Path $logdir 'run_stdout.log'
$stderr = Join-Path $logdir 'run_stderr.log'
$Host.UI.RawUI.WindowTitle = 'Mortal 天凤采集进度'
while ($true) {
  Clear-Host
  $now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
  Write-Host ('=' * 62)
  Write-Host "  Mortal 天凤采集进度            $now"
  Write-Host ('=' * 62)
  $remain = 0
  $rate = 0
  try {
    $r = Import-Csv $csv
    $total = $r.Count
    $ok    = @($r | Where-Object { $_.downloaded -eq 'ok' }).Count
    $fail  = @($r | Where-Object { $_.downloaded -like 'fail:*' }).Count
    $conv  = @($r | Where-Object { $_.converted  -eq 'ok' }).Count
    $remain = $total - $ok
    $pct = if ($total -gt 0) { [math]::Round($ok * 100 / $total, 2) } else { 0 }
    Write-Host "清单总数     : $total"
    Write-Host "已下载       : $ok   ($pct%)"
    Write-Host "待下载       : $remain"
    Write-Host "下载失败行   : $fail"
    Write-Host "已转换 mjai  : $conv"
  } catch {
    Write-Host "读取 CSV 失败(可能正在写入): $($_.Exception.Message)"
  }
  Write-Host ('-' * 62)
  $line = Get-Content $stderr -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
  if ($line -match '(\d+)/(\d+)\s*\[([^<]+)<([^,]+),\s*([\d.]+)game/s') {
    $cur = $Matches[1]; $tot2 = $Matches[2]; $el = $Matches[3]; $eta = $Matches[4]
    $rate = [double]$Matches[5]
    Write-Host "本轮进度     : $cur / $tot2"
    Write-Host "当前速率     : $rate 局/秒"
    Write-Host "本轮已用时   : $el      本轮剩余: $eta"
    if ($rate -gt 0 -and $remain -gt 0) {
      $ts = [TimeSpan]::FromSeconds([int]($remain / $rate))
      Write-Host ("预计全部完成 : 约 {0} 小时 {1} 分" -f $ts.Hours, $ts.Minutes)
    }
  } else {
    Write-Host "本轮进度     : (暂无 tqdm 输出)"
    if ($line) { Write-Host "stderr 末行  : $line" }
  }
  Write-Host ('-' * 62)
  $p = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*tenhou_pipeline*' })
  if ($p.Count -gt 0) {
    Write-Host "采集进程     : 运行中 (PID $(( $p | ForEach-Object { $_.ProcessId }) -join ', '))"
  } else {
    Write-Host "采集进程     : 未运行 (已结束或已被终止)" -ForegroundColor Yellow
  }
  $lastout = Get-Content $stdout -Tail 1 -Encoding UTF8 -ErrorAction SilentlyContinue
  Write-Host "最近关键日志 : $lastout"
  Write-Host ''
  Write-Host '(每 30 秒自动刷新；关闭本窗口即可退出)' -ForegroundColor DarkGray
  Start-Sleep -Seconds 30
}