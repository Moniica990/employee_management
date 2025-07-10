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

Sure! Here's the modified version of the **"🚀 How to Run"** and **"📂 Folder Structure"** sections for your repo: [https://github.com/Moniica990/employee\_management](https://github.com/Moniica990/employee_management)

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Moniica990/employee_management.git
cd employee_management

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv env
env\Scripts\activate      # On Windows
source env/bin/activate   # On macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python main.py
```

## 📂 Folder Structure

```
📁 employee_management/
├── 📁 dataset/
│   ├── 📁 employees/
│   └── 📁 customers/
├── 📁 logs/
│   └── attendance.csv
├── 📄 main.py
├── 📄 requirements.txt
```

Let me know if you'd like a `📺 Demo` or `📊 Features` section updated as well!


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



