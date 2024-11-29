# Airline Passenger Satisfaction Prediction
Predicting passenger satisfaction using KNN, Random Forest and MLP

In some parts of the world, passenger feedback plays a vital role in helping airlines and travel service providers improve their services. 🛫 In this project, we aim to develop a model that predicts passengers' overall sentiment—whether they are satisfied or dissatisfied—with the services provided by airlines.

This dataset also offers a great opportunity for data enthusiasts to explore further analysis and visualization, gaining valuable insights from the data. Whether you're interested in machine learning or data analysis, this project provides a versatile playground for enhancing your skills.

---

## Project Goals

- **Build a Predictive Model**: Use the dataset to predict passengers' satisfaction levels based on various features.
- **Data Exploration and Visualization**: Gain insights from the data by performing in-depth analysis and creating meaningful visualizations.

---

## Features of the Dataset

The dataset contains multiple features related to passengers' flight experiences, including:

<center>
<div dir=rtl style="direction: rtl;line-height:200%;font-family:vazir;font-size:medium">
<font face="vazir" size=3>
| <b>ستون</b> | <b>توضیحات</b> |
| :---: | :---: |
| <code>Gender</code> | جنسیت مسافر |
| <code>Customer Type</code> | نوع مسافر (از نظر وفاداری) |
| <code>Age</code> | سن مسافر |
| <code>Type of Travel</code> | نوع سفر (کاری یا تفریحی) |
| <code>Class</code> | کلاس پرواز |
| <code>Flight Distance</code> | مسافت پرواز |
| <code>Inflight wifi service</code> | رضایت از امکانات وای‌فای در هنگام پرواز (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Departure/Arrival time convenient</code> | رضایت از زمان حرکت/رسیدن پرواز (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Ease of Online booking</code> | رضایت از راحتی رزرو آنلاین (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Gate location</code> | رضایت از موقعیت گِیت پرواز (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Food and drink</code> | رضایت از غذا و نوشیدنی‌های ارائه‌شده (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Online boarding</code> | رضایت از بوردینگ آنلاین (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Seat comfort</code> | رضایت از راحتی صندلی (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Inflight entertainment</code> | رضایت از امکانات سرگرمی در هنگام پرواز (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>On-board service</code> | رضایت از خدمات ارائه‌شده در هنگام پرواز (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Leg room service</code> | رضایت از فضای پا (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Baggage handling</code> | رضایت از خدمات باربری (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Checkin service</code> | رضایت از خدمات چک‌این (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Inflight service</code> | رضایت از خدمات ارائه‌شده در هنگام پرواز (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Cleanliness</code> | رضایت از تمیزی هواپیما (۰ به‌معنی عدم نظر - از ۱ تا ۵) |
| <code>Departure Delay in Minutes</code> | تاخیر حرکت (دقیقه) |
| <code>Arrival Delay in Minutes</code> | تاخیر رسیدن (دقیقه) |
| <code>satisfaction</code> | رضایت مسافران (رضایت: ‌<code>satisfied</code>، عدم رضایت یا خنثی: <code>neutral or dissatisfied</code>) |
</font>
</div>
</center>

---

## How to Use This Repository

1. **Data Preparation**: Load and preprocess the dataset for analysis and modeling.
2. **Exploratory Data Analysis (EDA)**: Use the provided scripts to explore and visualize the data, uncovering trends and correlations.
3. **Modeling**: Build and evaluate machine learning models to predict passenger satisfaction.
4. **Insights**: Extract actionable insights from the analysis and predictions.

---
