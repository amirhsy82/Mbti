from MBTI_app import Message

messages = Message.query.order_by(Message.id).all()
num_messages = Message.query.count()

type_dict = {}
for message in messages:
    if message.mbti_type not in type_dict.keys():
        type_dict.__setitem__(message.mbti_type, 1)
    else:
        type_dict[message.mbti_type] += 1

print(type_dict)
