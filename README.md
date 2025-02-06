# **Appointment or Disappointment - AI-Based Train Delay Prediction**  

This project is part of the course **"Advanced AI-Based Application Systems"**, taught by **Prof. Dr.-Ing. Marcus Grum** at the University of Potsdam. The goal is to predict train delays in the Berlin and Brandenburg regions using AI models. We have implemented both an **Artificial Neural Network (ANN)** and an **Ordinary Least Squares (OLS)** regression model to tackle this problem.  

---

## **📁 Project Structure**  
```
├── code/                      # Python scripts for data processing and modeling
│   ├── dataPreparation/       # Data cleaning and preprocessing scripts
│   │   ├── clean_train_data.py
│   │   ├── clean_weather_data.py
│   │   ├── fetch_trains.py        # Fetches train data from Kaggle
│   │   ├── match_state_to_station.py
│   │   ├── merge_and_split_data.py
│   │   ├── prep_data.py
│   │   └── scrape_weather.py      # Scrapes weather data from DWD (Unused)
│   ├── model/                 # Trained models and evaluation results
│   │   ├── ANN/                # ANN metrics and visualizations
│   │   └── OLS/                # OLS metrics and visualizations
│   ├── ANN.py                   # ANN model implementation
│   ├── applyANN.py               # Script to apply trained ANN model
│   ├── applyOLS.py               # Script to apply trained OLS model
│   ├── feature_selection.py      # Feature selection and importance analysis
│   └── OLS_model.py              # OLS model implementation
├── data/                        # Dataset storage
│   ├── cleaned/                 # Cleaned and structured datasets
│   └── scraped/                 # scraped data
│       ├── DBtrainrides.csv      # Large file, it is not included in the git, please download from the provided link below
│       ├── scraped_data.csv      # Train data from Kaggle
│       └── scraped_weather.csv   # Unused
├── images/                      # Docker images for deployment
│   ├── activationBase_appointmentOrDisappointment/
│   │   ├── Dockerfile
│   │   └── README.md
│   ├── codeBase_appointmentOrDisappointment/
│   │   ├── Dockerfile
│   │   └── README.md
│   ├── knowledgeBase_appointmentOrDisappointment/
│   │   ├── Dockerfile
│   │   └── README.md
│   ├── learningBase_appointmentOrDisappointment/
│   │   ├── Dockerfile
│   │   └── README.md
│   └── docker-compose.yml      # Compose file for training and evaluation
├── scenarios/                 # Use case scenarios for model application
│   ├── ANN/
│   │   └── docker-compose.yml  # Compose file for ANN application
│   └── OLS/
│       └── docker-compose.yml  # Compose file for OLS application
├── .gitignore
└── README.md
```

---

## **Data Sources**  

- **Train Delays Data (Kaggle)**  
  - **Dataset 1**: Downloaded using `fetch_trains.py`, available on [Kaggle](https://www.kaggle.com/datasets/nokkyu/deutsche-bahn-db-delays/data/?select=DBtrainrides.csv).  
  - **Dataset 2**: Manually downloaded (not included in Git due to its large size, >700MB). You can download the dataset [here](https://drive.google.com/drive/folders/1j-b6GL-Ng2o3Ge0tsPiIoIOGq100NpsQ).  

- **Weather Data**  
  - **Scraped from DWD (Deutscher Wetterdienst)** but not used due to mismatched timestamps with train data. See dataset [here](https://opendata.dwd.de/climate_environment/CDC/event_catalogues/germany/precipitation/CatRaRE_v2024.01/data/).  
  - **Meteostat Library**: Used instead for weather feature integration.  

---

## **Data Preparation**  

The data processing workflow includes:  
- **Cleaning & Normalization**: Handling missing values and inconsistencies.  
- **Feature Selection**: Identifying relevant features for training.  
- **Filtering Data**: Selecting only **Berlin & Brandenburg** regions.  
- **Data Splitting**: 80% training, 20% testing.  

---

## **🧠 Models**  

### **Artificial Neural Network (ANN)**  
- Implemented using **TensorFlow**.  
- Includes **feature scaling, batch normalization, and dropout layers**.  
- Optimized using **Mean Squared Error (MSE)** and **Mean Absolute Error (MAE)**.  

### **Ordinary Least Squares (OLS)**  
- Implemented using **Statsmodels**.  
- Serves as a **baseline regression model**.  
- Evaluated using **residual analysis, Q-Q plots, and regression diagnostics**.  

---

## **Docker Setup**  

The project is containerized for reproducibility and deployment. It includes four Docker images:  

| Image Name         | Purpose |
|--------------------|---------|
| **learningBase**   | Stores training and testing data. |
| **activationBase** | Stores activation data. |
| **knowledgeBase**  | Stores trained ANN and OLS models. |
| **codeBase**       | Contains scripts for applying models. |

Each image has a `README.md` explaining its setup, and all images are based on the `busybox` image.


This project uses Docker for containerization, ensuring reproducible results. Follow these steps to set up the environment and run the models:

### Setting Up the Environment

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/luisahorlledecastro/AI-CPS.git
   cd AI-CPS
   ```

2. **Create a Docker Volume:**

   This volume will be used to share data between the containers and your local machine.

   ```bash
   docker volume create ai_system
   ```

3. **Build the Docker Images:**

   Build the images for training and prediction:

   ```bash
   docker-compose build
   ```

   This command creates the necessary Docker images.

### Running the Models

#### Training and Evaluation (Initial Setup):

The initial training and evaluation of the models are performed within a Docker container.

1. **Start the Training Container:**

   ```bash
   docker-compose up -d  # Runs in detached mode
   ```

2. **Navigate to the Code Directory:**

   ```bash
   cd <your_path>/AI-CPS/code
   ```

3. **Run the Training Scripts:**

   ```bash
   python ANN.py
   python OLS_model.py
   ```

   The trained models and results are saved to the `model/` directory on your local machine, which is mounted as a volume.

4. **Exit the container:**

   ```bash
   exit
   ```

5. **Stop the Training Container:**

   ```bash
   docker-compose down
   ```

#### Applying the Trained Models (Inference):

After training the models, you can apply them to new data using the separate `docker-compose.yml` files in the `scenarios` directory.

**Applying the ANN Model:**

1. **Navigate to the ANN Scenario Directory:**

   ```bash
   cd scenarios/ANN
   ```

2. **Run the ANN Application:**

   ```bash
   docker-compose up -d
   ```

**Applying the OLS Model:**

1. **Navigate to the OLS Scenario Directory:**

   ```bash
   cd scenarios/OLS
   ```

2. **Run the OLS Application:**

   ```bash
   docker-compose up -d
   ```

These commands will start the necessary containers, using the `knowledgeBase`, `activationBase`, and `codeBase` images. The results of the inference are saved to your local machine via the mounted volume.

### Accessing Results

Model evaluation metrics and visualizations are stored in `model_metrics/`, inside the respective model directories (`ANN/` or `OLS/`).  




### **Docker Images**
The following images are published on **Docker Hub** and can be pulled directly:

```bash
docker pull luisahorlledecastro/learningBase_appointmentOrDisappointment:latest
docker pull luisahorlledecastro/activationBase_appointmentOrDisappointment:latest
docker pull luisahorlledecastro/knowledgeBase_appointmentOrDisappointment:latest
docker pull luisahorlledecastro/codeBase_appointmentOrDisappointment:latest
```


---

## **⚙️ Dependencies**  

- **Python**: 3.12.4  
- **Libraries**:  
  - TensorFlow  
  - Statsmodels  
  - Pandas, NumPy, Scikit-learn  
  - BeautifulSoup (for web scraping)  
  - Meteostat (for weather data)  
  - Matplotlib, Seaborn (for visualization)  


---

## Authors
This project was collaboratively developed by:
- Dita Pelaj
- Luísa Hörlle de Castro

---

## License
This project is licensed under the AGPL-3.0 License as required by the course.
