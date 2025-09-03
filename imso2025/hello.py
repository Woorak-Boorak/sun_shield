def say_hello(name):
    """이름을 받아 인사말을 출력하는 함수"""
    print(f"Hello, {name}! Welcome to Jetson Nano.")

if __name__ == "__main__":
    my_name = "imso"
    say_hello(my_name)