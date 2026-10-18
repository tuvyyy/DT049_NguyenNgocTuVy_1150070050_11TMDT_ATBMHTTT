"""Native button navigation; no radio DOM manipulation or browser-specific CSS."""

import streamlit as st


def _select(key, option):
    st.session_state[key] = option


def navigation(options, *, key):
    if st.session_state.get(key) not in options:
        st.session_state[key] = options[0]
    with st.container(key=f"{key}_nav"):
        columns = st.columns(len(options), gap="small")
        for index, (column, option) in enumerate(zip(columns, options)):
            with column:
                st.button(option, key=f"{key}_{index}", width="stretch",
                          type="primary" if st.session_state[key] == option else "secondary",
                          on_click=_select, args=(key, option))
    return st.session_state[key]
