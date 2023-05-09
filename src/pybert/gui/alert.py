from traitsui.message import error, message

def error_popup(content):
    error(content, title="PyBERT Alert")

def message_popup(content):
    message(content, title="PyBERT")
