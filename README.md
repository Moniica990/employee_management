# Employee Management System with Computer Vision

This project uses **Computer Vision** to distinguish between **customers** and **employees** through facial recognition, streamlining attendance and access control in a workplace environment.

## 🔍 Key Features

* 🎯 **Facial Recognition**: Detect and classify faces as either *employee* or *customer* using pre-trained models.
* 🕒 **Automated Attendance**: Automatically logs employee check-ins and check-outs.
* 🧾 **Real-Time Logging**: Records attendance data into a CSV or database.
* 🧑‍💼 **Employee Management Dashboard**: Add, update, or remove employee records easily.
* 👥 **Customer Detection**: Recognizes and ignores non-employee (customer) faces.

## 💡 Tech Stack

* Python
* OpenCV & face\_recognition
* Streamlit / Flask (for UI)
* SQLite / CSV (for data storage)

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/yourname/employee-management-vision.git
cd employee-management-vision

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python main.py
```

## 📂 Folder Structure

```
📁 dataset/
    ├── employees/
    └── customers/
📁 logs/
    └── attendance.csv
main.py
requirements.txt
```

## 📸 How It Works

1. Capture face via webcam.
2. Match it against the trained database.
3. Log employee attendance or ignore if customer.
4. Update dashboard in real-time.

## 📌 Use Cases

* Offices and corporate spaces
* Co-working spaces
* Retail stores with staff access control


### 📺 computer vision model

![image](https://github.com/user-attachments/assets/f2052e61-1c6a-4841-80d7-a97472109ae6)



### 📽️ Working Demo

https://github.com/user-attachments/assets/1cbee21e-4f65-4422-a8bb-fbf691d11e38



