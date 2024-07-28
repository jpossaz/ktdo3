import time
import ktdo3
import random
import string


def random_string(length):
    return "".join(random.choice(string.ascii_letters) for i in range(length))


if __name__ == "__main__":
    string_pool = [random_string(10) for i in range(100)]
    print("Initializing...")
    a = []
    for i in range(5000):
        num_tags = random.randint(0, 50)
        tags = random.sample(string_pool, num_tags)
        a.append(set(tags))
    b = []
    for i in range(5000):
        num_tags = random.randint(0, 50)
        tags = random.sample(string_pool, num_tags)
        b.append(set(tags))

    print("Starting...")
    start = time.time()
    ktd = ktdo3.kernel_tag_distance(a, b)
    print(time.time() - start, ktd)
