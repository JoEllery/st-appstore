{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN+YMxGJ04ZzIzL6iq8g02W",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/JoEllery/st-appstore/blob/main/AppStoreHome.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RN7J7F2lcX4z"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "\n",
        "st.set_page_config(page_title=\"App Store Analytics\")\n",
        "st.write(\"Use this resource to understand how sentiment is evolving on the Apple App Store.\")\n",
        "\n",
        "class AppBot:\n",
        "\n",
        "  def main():\n",
        "\n",
        "    appname = st.text_input(\"App name:\")\n",
        "    r_num = st.text_input(\"Review number:\", placeholder=\"Reccomended: 200\")\n",
        "    min_date = st.text_input(\"Start date:\", \"2022-01-01\")\n",
        "\n",
        "    if appname and r_num and min_date:\n",
        "\n",
        "      st.write(\"you entered things!\")\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "\n",
        "  obj = AppBot()\n",
        "  AppBot.main()"
      ]
    }
  ]
}