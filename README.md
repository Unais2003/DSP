## Digital Signal Processing (DAT_SS2026_2)

[![GitLab](https://img.shields.io/badge/GitLab-Project-orange?style=flat&logo=gitlab)](https://git-iit.fh-joanneum.at/u58e87)
[![Status](https://img.shields.io/badge/Status-In_Progress-blue?style=flat)](#)



## Overview

This repository contains comprehensive coursework for **Digital Signal Processing (DAT_SS2026_2)** at FH Joanneum. The project features an interactive Streamlit data application that consolidates all course assignments and practical exercises.



##  Author

- **Name:** Perinchikkal Muhammed Unais
- **Institution:** FH Joanneum
- **Program:** Data Science and Artificial Intelligence
- **Course Code:** DAT_SS2026_2
- **Creation Date:** 14 March 2026
- **Last Updated:** 18 April 2026

---

##  Project Description

This project implements various Digital Signal Processing concepts and techniques through a Streamlit-based interactive application. It covers fundamental DSP principles, algorithms, and their practical applications.

##  Core Topics Covered

- Sampling and Quantization
- Fourier Analysis & FFT
- Filtering Techniques
- Signal Convolution
- Z-Transform
- Digital Filter Design
- Signal Modulation & Demodulation
- Spectral Analysis


---

##  Project Structure

```
DSP/
├── README.md                 # Project documentation
├── project.toml          # Python dependencies
├── Home.py               # Main Streamlit application
├── uv.lock                 
└── data/              # data folder
    ├── assignment_01/
    ├── assignment_02/
    └── ...
└── pages/              # different pages
    ├── assignment_01/
    ├── assignment_02/
    └── ...
└── resources/              # external resources
└── src/                  # modules and func
    ├── assignment_01/
    ├── assignment_02/
    └── ...
```

---

##  Technologies & Tools

- **Python 3.8+**
- **Streamlit** - Interactive web application framework
- **NumPy** - Numerical computing
- **Pandas** - Data manipulation and analysis
- **Matplotlib & Plotly** - Data visualization
- **SciPy** - Scientific computing and signal processing

---

##  Getting Started

### Prerequisites

Ensure you have Python 3.8 or higher installed on your system.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Unais2003/DSP.git
   cd DSP
   ```

2. **Create a virtual environment:**
   ```bash
   uv sync
   ```


### Running the Application

To launch the Streamlit application:

```bash
uv run streamlit run  home.py    
```

The application will open in your default web browser at `http://localhost:8501`

---






