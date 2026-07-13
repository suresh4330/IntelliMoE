# AI Model Benchmarking Report: IntelliMoE Providers

This report evaluates and compares API performance, latency, token throughput, and estimated hosting costs across model providers (**Groq** and **Google Gemini**).

---

## 📈 Performance Summary Comparison

| Provider (Model) | Avg Response Time | First Token Latency | Total Response Time | Total Tokens | Success Rate | Est. Cost |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Groq (llama3-8b-8192)** | 0.452s | 0.054s | 3.17s | 1,400 | 100.0% | $0.00024 |
| **Gemini (gemini-1.5-flash)** | 0.921s | 0.138s | 6.45s | 1,430 | 100.0% | $0.00033 |

---

## 🏆 Benchmark Leaderboard Leaders

- ⚡ **Fastest Model (Response Time)**: **Groq (llama3-8b-8192)**
- ⏱️ **Lowest First-Token Latency**: **Groq (llama3-8b-8192)**
- ✅ **Highest Success Rate**: **Tie - Both Groq and Gemini (100.0%)**
- 💵 **Lowest Estimated Cost**: **Groq (llama3-8b-8192)**

---

## 🥇 Overall Provider Winner

Based on a weighted score combining response times, success flags, and query pricing:
- 🏆 **Overall Benchmark Winner**: **Groq (llama3-8b-8192)**

---

## 🗄️ Historical Runs Summary

- **Total Historical Runs Logged**: 2
- **History Database Path**: `data/benchmark_history.csv`
- **Report Generation Timestamp**: 2026-07-12 01:31:00
