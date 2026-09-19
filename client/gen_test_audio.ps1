# Generates Chinese TTS wav files for ASR testing (16kHz 16bit mono).
# Part of A2 joint-testing samples (contract docs/API_CONTRACT.md A.11).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File client/gen_test_audio.ps1
#   # optional: -CmdFile <utf8 text, one command per line> -OutDir <wav output dir>
#
# Why .ps1: Windows System.Speech TTS is built-in (Microsoft Huihui zh-CN),
# zero Python dependencies. Script body is pure ASCII to avoid GBK garbling;
# Chinese commands live in test_audio_commands.txt (UTF-8), read with -Encoding UTF8.
param(
    [string]$CmdFile = "$PSScriptRoot\test_audio_commands.txt",
    [string]$OutDir = "D:\EgoMed-Agent\data\samples\audio"
)

Add-Type -AssemblyName System.Speech

if (-not (Test-Path $OutDir)) { New-Item -ItemType Directory -Path $OutDir -Force | Out-Null }

$lines = Get-Content -Path $CmdFile -Encoding UTF8 | Where-Object { $_.Trim().Length -gt 0 }

$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)

$i = 0
foreach ($line in $lines) {
    $i++
    $name = "cmd_{0:d2}.wav" -f $i
    $path = Join-Path $OutDir $name
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    $s.SelectVoice("Microsoft Huihui Desktop")
    $s.SetOutputToWaveFile($path, $fmt)
    $s.Speak($line.Trim())
    $s.Dispose()
    Write-Output ("OK {0}" -f $name)
}
Write-Output ("DONE {0} files -> {1}" -f $i, $OutDir)
