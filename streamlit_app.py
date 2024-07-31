import os
import pandas as pd
from openai import OpenAI
from sqlalchemy import create_engine
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import bokeh as bk
import pygal as pg

# Initialize global variables
client = None             # OpenAI client for database queries
viz_client = None         # OpenAI client for visualization 
engine  = None            # Database engine
sys_prompt  = None        # System prompt for database queries
viz_prompt = None         # System prompt for visualization
messages = None           # Messages for database queries
viz_messages = None       # Messages for visualization
db_name = None            # Database name

password = os.getenv("MYSQL_ROOT_PASSWORD")  # Get the password from environment variable


# Get OpenAI client
def get_openai_client():
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API_KEY environment variable not set")
    return OpenAI(api_key=api_key)


# Connect to the database
def get_database_engine():
    db_url = f'mysql+mysqldb://root:{password}@localhost:3306/{db_name}'
    return create_engine(db_url)


# Getting database schema and table descriptions to initialize the assistant 
def initialize_database():
    engine = get_database_engine()

    tables_db = pd.read_sql_query("SHOW TABLES;", engine)
    table_names = list(tables_db[f'Tables_in_{db_name}'])
    tables = []

    for table in table_names:
        table_desc = pd.read_sql_query(f"DESCRIBE {table};", engine)
        tables.append(f"""{table} Table - '''{table_desc}'''""")

    prompt = f"""You are a SQL database administrator. You are supposed to convert user prompt to SQL query that will help the user to answer any question. You are only allowed to respond to database/table related prompts. The name of the database is "{db_name}" and here's the table descriptions - {tables}. You should only output the query without any prefix or anything extra added to the query. Also, try to be helpful by framing the query in a way that clearly expresses the information to the user. And if the query is not related to the database, you sould respond accordingly."""
    
    return prompt   # return the system prompt


# Get the system prompt for visualization
def get_viz_prompt_system():
    sys_prompt = """you will be given a dataframe and your task is to visually represent the data. You have imported matplotlib.pyplot as plt, seaborn as sns, bokeh as bk and pygal as pg. You can use these packages to plot the data. You can also use pandas to manipulate the data. Plot the data in unique ways, you can use any type of plots which are available in the included libraries. The output for visualization should only contain python executable code in a single string. Don't include the first word as 'python' in the output, directly output executable code. If the given dataframe is not plotable, then return a message saying that the data is not visualizable."""
    return sys_prompt


# Get the visualization prompt
def get_viz_prompt(df, plot_type, x_col=None, y_col=None):
    if df is not None and not df.empty:
        if x_col and y_col:
            return f"""I have a dataframe df with the following data: {df.to_dict()}\nI want to visually represent this data in form of a {plot_type} with {x_col} as the x-axis and {y_col} as the y-axis. Use the packages that I have imported and plot the data. Only return the python code to plot the data. And if the given dataframe is not visualizable, you should output accordingly."""
        else:
            return f"""I have a dataframe df with the following data: {df.to_dict()}\nI want to visually represent this data in form of a {plot_type}. Use the packages that I have imported and plot the data. Only return the python code to plot the data. And if the given dataframe is not visualizable, you should output accordingly."""
    else:
        return None
    

# Keeps the conversation going with the assistant
def conversation(client, messages, prompt):
    if prompt is not None:
        messages.append({"role": "user", "content": prompt})   # Append user prompt to messages to keep track of the conversation
    
        chat_completion_stream = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0.2
        )
        response_text = chat_completion_stream.choices[0].message.content
        return response_text
    else:
        return None


# Initialize messages with system prompt for the conversation
def initialize_messages(system_prompt):
    return [{"role": "system", "content": system_prompt}]


# user input
def get_user_input():
    return st.text_input("User:", key="user_input")


# Display the response from the assistant
def display_response(response):
    with st.expander("See SQL Query"):
        st.code(f"Assistant:\n{response}")


# Execute the query and display the result
def display_query_result(engine, query):
    try:
        df = pd.read_sql_query(query, engine)
        if df.empty:
            st.write("The query executed successfully but returned no results.")
            return None
        else:
            return df
    except:
        return None


# Initializing global variables
def initialize_utility_functions():
    global client, viz_client, engine, messages, viz_messages, sys_prompt, viz_prompt

    client = get_openai_client()
    viz_client = get_openai_client()
    engine = get_database_engine()
    sys_prompt = initialize_database()
    viz_prompt = get_viz_prompt_system()
    messages = initialize_messages(sys_prompt)
    viz_messages = initialize_messages(viz_prompt)


def get_db_name_and_user_input():
    global db_name
    st.sidebar.title("SQL Database Assistant")
    db_name = st.sidebar.text_input("Enter the database name:", "sales", key="db_name")
    return st.sidebar.text_input("User:", key="user_input")


def get_df_output(user_prompt):
    try:
        initialize_utility_functions()  
        response = conversation(client, messages, user_prompt)
        if response is None:
            return None
          
        display_response(response)
        try:
            df = display_query_result(engine, response)
            if df is not None:
                st.dataframe(df)
                return df

        except Exception as e:
            st.write(f"An unexpected error occurred: {e}")
            return None
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")
        return None
    

def get_viz_output(df):
    if df is not None and not df.empty:
        auto_plot = st.checkbox("Auto Plot", value=True)

        if not auto_plot:
            x_col = st.selectbox("Select X-axis", df.columns, key="x_axis")
            y_col = st.selectbox("Select Y-axis", df.columns, key="y_axis")
        else:
            x_col = y_col = None

        selection = st.selectbox("Plot Type", ("None", "Bar plot", "Pie plot", "Histogram", "Line plot", "Scatter plot"), placeholder="Graph type...")

        if st.button("Visualize") and selection != "None":
            prompt = get_viz_prompt(df, selection, x_col, y_col)
            viz_response = conversation(viz_client, viz_messages, prompt)
            try:
                viz_response = viz_response.replace("plt.show()", "st.pyplot(plt)")
            except:
                pass     
            with st.expander("See Plot Code"):
                st.code(viz_response)
            try:
                if "plt.title('Missing Values in DataFrame')" not in viz_response:
                    exec(viz_response)
            except:
                pass


def main():
    user_prompt = get_db_name_and_user_input()

    if "clicked" not in st.session_state:
        st.session_state["clicked"] = False

    if st.sidebar.button("Start") or st.session_state["clicked"]:
        st.session_state["clicked"] = True
        if db_name and user_prompt:
            df = get_df_output(user_prompt)
            get_viz_output(df)


if __name__ == "__main__":
    main()
