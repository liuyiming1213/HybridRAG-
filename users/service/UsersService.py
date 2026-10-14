import random
from email.mime.text import MIMEText
import smtplib
from users.dao import UsersDao
import os
from common import RedisUtil
from common import JWTUtil   # 导入JWT工具，用于登录成功后生成Token
from dotenv import load_dotenv

load_dotenv()

#发送邮件
def send_email(email):
    #验证是否注册过
    flag = False
    #isinstance(a,b)返回布尔值，判定a是否是b的实例
    if isinstance(UsersDao.check_email(email),list):
        flag = True
    #2.根据结果判定是否需要发邮件
    if not flag:
        #不发邮件
        return {
            "code":500,
            "msg":"邮箱账号不存在",
            "data":None
        }
    #发邮件
    #随机生成4位验证码
    code = ""
    for i in range(4):
        #随机生成一个0-9的数字
        code += str(random.randint(0,9))
    #配置发送信息
    sender = os.getenv("SENDER_EMAIL")  # 发送方
    sender_pwd = os.getenv("SENDER_EMAIL_PASSWORD")  # 发送方授权码
    subject = "你的验证码来咯~~~"  # 邮件主题
    content = f"你的验证码为：{code}, 过期时间为：60s"
    #创建邮件对象---写发送邮件的信息在这个对象里
    message = MIMEText(content,"plain","utf-8")
    #添加内容在message中
    message["From"] = sender
    message["To"] = email
    message["Subject"] = subject
    try:
        # 构建发送邮件的对象
        smtp = smtplib.SMTP(
            host=os.getenv("SMTP_HOST"),
            port=int(os.getenv("SMTP_PORT"))
        )
        # 开启邮件发送服务 TLS
        smtp.starttls()
        # 验证发送方和发送方的授权码是否能对上
        smtp.login(sender, sender_pwd)
        # 发送邮件 -- sendmail 发送邮件的方法：发送方，接收方，邮件对象
        smtp.sendmail(sender, email, message.as_string())
        # 发送成功后退出
        smtp.quit()
        # 把验证码存入到redis中
        r = RedisUtil.get_redis_conn()
        # redis存入数据通过key,value格式 --- 取值通过key
        r.setex(email, 60, code)  # 60秒后过期，key:email value:code
        # 关闭连接
        RedisUtil.close_redis_conn(r)
        return {
            "code": 200,
            "msg": "发送成功",
            "data": None,
        }
    except Exception as e:
        print(f"邮件发送失败：{e}")
        return {
            "code": 500,
            "msg": "发送失败",
            "data": None,
        }

# send_email 函数结束

#  验证码登录
#  流程：用户输入邮箱 → 后端发验证码到邮箱 → 用户填入验证码 →
# 后端从Redis校验 → 通过后查询用户信息 → 签发JWT Token
def check_code(checkCodeEntity):
    # 取出用户传递过来的邮箱号和验证码
    email = checkCodeEntity.email
    code = checkCodeEntity.code
    # 通过用户输入的邮箱号，取出redis中存入的验证码
    r = RedisUtil.get_redis_conn()
    redis_code = r.get(email)
    RedisUtil.close_redis_conn(r)
    # 判断验证码是否过期
    if redis_code is None:
        return {"code": 500, "msg": "验证码已过期", "data": None}
    # 进行验证码的比对
    if redis_code != code:
        return {"code": 500, "msg": "验证码不正确", "data": None}
    # 验证通过，查询用户信息
    user_info = UsersDao.check_email(email)
    user_id = user_info[0]["id"] if user_info else None
    username = user_info[0]["username"] if user_info else ""
    if user_id:
        # 签发 JWT Token
        return {
            "code": 200,
            "msg": "验证码正确，登录成功",
            "data": {
                "access_token": JWTUtil.create_access_token(user_id, username),
                "refresh_token": JWTUtil.create_refresh_token(user_id, username),
                "username": username,
            },
        }
    return {"code": 500, "msg": "用户信息异常", "data": None}
