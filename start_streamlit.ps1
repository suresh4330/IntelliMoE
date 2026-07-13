$env:PYTHONUTF8 = "1"
Set-Location "C:\Users\annam\OneDrive\Desktop\IntelliMoE"
& "C:\Users\annam\AppData\Local\Programs\Python\Python314\python.exe" -m streamlit run "ui/app.py" --server.port 8501 --server.address localhost --server.headless true *>&1 | Tee-Object -FilePath "streamlit.log"
