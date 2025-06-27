import streamlit as st
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import pandas as pd
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import plotly.express as px
import plotly.graph_objects as go

# --- File paths and Setup ---
ATTENDANCE_PATH = "Attendance.csv"
TRAINING_PATH = "Training_images"

os.makedirs(TRAINING_PATH, exist_ok=True)
if not os.path.exists(ATTENDANCE_PATH) or os.stat(ATTENDANCE_PATH).st_size == 0:
    with open(ATTENDANCE_PATH, "w") as f:
        f.write("Name,Time\n")

@st.cache_resource
def load_known_faces():
    images = []
    classNames = []
    if os.path.exists(TRAINING_PATH):
        for cl in os.listdir(TRAINING_PATH):
            curImg = cv2.imread(f"{TRAINING_PATH}/{cl}")
            if curImg is not None:
                images.append(curImg)
                classNames.append(os.path.splitext(cl)[0])
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList, classNames

def mark_attendance(name):
    now = datetime.now()
    dtString = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_PATH, "r+") as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            f.writelines(f"\n{name},{dtString}")
            return True
    return False

class FaceRecognitionTransformer(VideoTransformerBase):
    def __init__(self):
        self.known_encodings, self.known_names = load_known_faces()
        self.last_attendance = {}
        self.frame_count = 0

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1
        if self.frame_count % 2 != 0:
            return img
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(self.known_encodings, encodeFace)
            faceDis = face_recognition.face_distance(self.known_encodings, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = self.known_names[matchIndex].upper()
                if (name not in self.last_attendance or 
                   (time.time() - self.last_attendance[name]) > 5):
                    if mark_attendance(name):
                        self.last_attendance[name] = time.time()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        return img

def calculate_hours():
    try:
        df = pd.read_csv(ATTENDANCE_PATH, parse_dates=["Time"])
        if df.empty:
            return pd.DataFrame()
        df["Date"] = df["Time"].dt.date
        daily_hours = df.groupby(["Name", "Date"])["Time"].agg(["min", "max"])
        daily_hours["Hours"] = (daily_hours["max"] - daily_hours["min"]).dt.total_seconds() / 3600
        total_hours = daily_hours.groupby("Name")["Hours"].sum().reset_index()
        return total_hours
    except:
        return pd.DataFrame()

st.set_page_config(page_title="Retail Employment Management", layout="wide")
st.title("Retail Employee Management System")

if "employees" not in st.session_state:
    st.session_state.employees = pd.DataFrame(columns=["Name", "Hourly Rate"])

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Attendance Capture",
    "📝 Employee Data",
    "📊 Hours Analysis",
    "💹 Profit Analysis",
    "🕵️‍♂️ Theft Detection"
])

# --- Tab 1: Attendance Capture ---
with tab1:
    st.header("Real-time Attendance Tracking")
    st.info("Position your face in the camera to register attendance")
    ctx = webrtc_streamer(
        key="face-recognition",
        video_transformer_factory=FaceRecognitionTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.subheader("Attendance Log")
    try:
        attendance_df = pd.read_csv(ATTENDANCE_PATH)
        if not attendance_df.empty:
            st.dataframe(attendance_df.tail(10), height=300)
        else:
            st.warning("No attendance records found")
    except Exception:
        st.warning("No attendance records found")

# --- Tab 2: Employee Data ---
with tab2:
    st.header("Employee Management")
    with st.form("employee_form"):
        col1, col2 = st.columns(2)
        name = col1.text_input("Employee Name")
        hourly_rate = col2.number_input("Hourly Rate ($)", min_value=0.0, value=15.0)
        image_file = st.file_uploader("Upload Employee Photo", type=["jpg", "png"])
        submitted = st.form_submit_button("Add Employee")
        if submitted and name:
            new_employee = pd.DataFrame([[name, hourly_rate]], columns=["Name", "Hourly Rate"])
            st.session_state.employees = pd.concat([st.session_state.employees, new_employee], ignore_index=True)
            if image_file:
                with open(os.path.join(TRAINING_PATH, f"{name}.jpg"), "wb") as f:
                    f.write(image_file.getbuffer())
                st.success(f"Employee {name} added successfully!")
                st.cache_resource.clear()
            else:
                st.warning("Employee added without photo - face recognition won't work")
    st.subheader("Current Employees")
    st.dataframe(st.session_state.employees, height=300, hide_index=True)
    if not st.session_state.employees.empty:
        to_delete = st.selectbox("Select employee to delete", st.session_state.employees["Name"])
        if st.button("Delete Employee"):
            st.session_state.employees = st.session_state.employees[st.session_state.employees["Name"] != to_delete]
            img_path = os.path.join(TRAINING_PATH, f"{to_delete}.jpg")
            if os.path.exists(img_path):
                os.remove(img_path)
            st.cache_resource.clear()
            st.success(f"Employee {to_delete} removed")

# --- Tab 3: Hours Analysis ---
with tab3:
    st.header("Employee Hours Analysis")
    # Example data simulating hours worked per employee
    attendance_data = {
        'Name': ['Alex Johnson', 'Samira Khan', 'James Wilson', 'Moniica'],
        'Hours': [120, 95, 110, 130]
    }
    hours_df = pd.DataFrame(attendance_data)

    # --- Pie Chart ---
    st.subheader("Proportion of Total Hours Worked")
    professional_colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA']
    fig_pie = px.pie(
        hours_df,
        values='Hours',
        names='Name',
        title='Total Hours Worked by Employee',
        color_discrete_sequence=professional_colors
    )
    fig_pie.update_traces(
        textposition='inside',
        textinfo='percent+label',
        pull=[0.03]*len(hours_df),
        marker=dict(line=dict(color='#111111', width=2))  # dark border for dark bg
    )
    fig_pie.update_layout(
        font=dict(family="Segoe UI, Arial", size=16, color="#FFF"),
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(color="#FFF")),
        paper_bgcolor='rgba(0,0,0,0)',  # transparent for black bg
        plot_bgcolor='rgba(0,0,0,0)',
        title_font_color='#FFF'
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # --- Line Chart with Arrows and Percentage Change ---
    st.subheader("Hours Worked with Percentage Change")
    pct_change = [0]
    for i in range(1, len(hours_df)):
        prev = hours_df.loc[i-1, 'Hours']
        curr = hours_df.loc[i, 'Hours']
        change = ((curr - prev) / prev) * 100
        pct_change.append(change)

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(
        x=hours_df['Name'],
        y=hours_df['Hours'],
        mode='lines+markers',
        name='Hours Worked',
        line=dict(color='#636EFA', width=3),
        marker=dict(size=12, color='#636EFA', line=dict(width=2, color='#111111'))
    ))
    for i in range(1, len(hours_df)):
        if pct_change[i] > 0:
            arrow_symbol = '⬆️'
            arrow_color = 'lime'
        elif pct_change[i] < 0:
            arrow_symbol = '⬇️'
            arrow_color = 'red'
        else:
            arrow_symbol = ''
            arrow_color = 'gray'
        fig_line.add_annotation(
            x=hours_df.loc[i, 'Name'],
            y=hours_df.loc[i, 'Hours'],
            text=f"{arrow_symbol} {pct_change[i]:.1f}%",
            showarrow=False,
            font=dict(color=arrow_color, size=16, family="Segoe UI, Arial"),
            yshift=18
        )
    fig_line.update_layout(
        title='Employee Hours Worked with Percentage Change',
        xaxis_title='Employee',
        yaxis_title='Hours Worked',
        template='plotly_dark',  # dark theme for black background
        hovermode='x unified',
        font=dict(family="Segoe UI, Arial", size=15, color="#FFF"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        title_font_color='#FFF',
        xaxis=dict(color='#FFF'),
        yaxis=dict(color='#FFF')
    )
    st.plotly_chart(fig_line, use_container_width=True)


# --- Tab 4: Profit Analysis ---
with tab4:
    st.header("Profit Analysis")
    st.write("This line graph shows profit with respect to the number of employees in your organization.")
    employee_counts = list(range(1, max(2, len(st.session_state.employees)) + 5))
    base_profit = 1000
    profit = [base_profit + 800 * n - 30 * n**2 for n in employee_counts]
    profit_df = pd.DataFrame({
        "Number of Employees": employee_counts,
        "Profit": profit
    })
    fig = px.line(
        profit_df,
        x="Number of Employees",
        y="Profit",
        markers=True,
        title="Profit vs. Number of Employees"
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(profit_df, hide_index=True)


    st.header("Sales/Performance Heatmap by Region and Product Category")
    regions = [
        'North America', 'Europe', 'Asia-Pacific', 
        'South America', 'Middle East', 'Africa'
    ]
    product_categories = [
        'Electronics',
        'Apparel',
        'Home & Kitchen',
        'Beauty & Personal Care',
        'Sports & Outdoors',
        'Toys & Games',
        'Grocery',
        'Automotive'
    ]
    np.random.seed(42)
    data = np.random.randint(1000, 10000, size=(len(regions), len(product_categories)))
    df = pd.DataFrame(data, index=regions, columns=product_categories)
    fig = px.imshow(
        df,
        text_auto=True,
        color_continuous_scale='Plasma',
        aspect='auto',
        labels=dict(x='Product Category', y='Region', color='Sales/Performance'),
        title='Sales/Performance by Region and Product Category'
    )
    fig.update_layout(
        font=dict(family='Segoe UI, Arial', size=14, color='#FFF'),
        paper_bgcolor='#111111',
        plot_bgcolor='#111111',
        xaxis=dict(tickangle=30, color='#FFF'),
        yaxis=dict(color='#FFF'),
        title_font_color='#FFF'
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Product Heatmap ---

with tab5:
    st.header("🚨 Theft Detection Dashboard")

    # Simulate theft detection (replace this with your real detection logic)
    if "theft_log" not in st.session_state:
        st.session_state.theft_log = []

    st.write("This dashboard will alert you if a theft is detected in the store.")

    # Demo button to simulate a theft event
    if st.button("Simulate Theft Event"):
        from datetime import datetime
        event_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.theft_log.insert(0, f"Theft detected at {event_time}")

    # If any theft event is in the log, show the latest as a dashboard alert
    if st.session_state.theft_log:
        st.error(f"🚨 {st.session_state.theft_log[0]}", icon="🚨")
    else:
        st.success("No theft detected. All clear! ✅")

    # Show theft log/history
    st.subheader("Theft Event Log")
    if st.session_state.theft_log:
        for i, event in enumerate(st.session_state.theft_log[:10]):
            st.warning(f"{i+1}. {event}")
    else:
        st.write("No theft events recorded.")



# --- Sidebar ---
with st.sidebar:
    st.header("Instructions")
    st.markdown("""
    1. **Attendance Capture**: 
        - Add employees first with photos
        - Face recognition auto-marks attendance
    2. **Employee Data**:
        - Add/remove employees
        - Upload employee photos
    3. **Hours Analysis**:
        - View hours worked (pie/line charts)
    4. **Profit Analysis**:
        - Visualize profit trend as employee count changes
    5. **Product Heatmap**:
        - See sales/performance by region and product
    """)
    st.download_button(
        "Download Attendance Data",
        data=open(ATTENDANCE_PATH, "rb").read(),
        file_name="attendance_data.csv"
    )

# --- CSS for clean UI ---
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { padding: 8px 20px; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { background-color: #f0f2f6; }
    [data-testid="stFileUploader"] { padding: 10px; border: 1px dashed #ccc; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)
