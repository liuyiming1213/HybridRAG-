#封装返回数据给客户请求的工具函数


#返回json数据
def response_json(code,msg,data):
    return {
        "code": code,
        "msg": msg,
        "data": data
    }
