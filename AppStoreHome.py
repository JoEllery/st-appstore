import streamlit as st

st.set_page_config(page_title="App Store Analytics")
st.write("Use this resource to understand how sentiment is evolving on the Apple App Store.")

class AppBot:

  def main():

    appname = st.text_input("App name:")
    r_num = st.text_input("Review number:", placeholder="Reccomended: 200")
    min_date = st.text_input("Start date:", "2022-01-01")

    if appname and r_num and min_date:

      st.write("you entered things!")





if __name__ == "__main__":

  obj = AppBot()
  AppBot.main()
