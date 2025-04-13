import random

choice_count = 0

def sort_three(nums):
    global choice_count
    choice_count += 1
    return sorted(nums) # 模拟用户总是选择从小到大

def compare_two(a, b, others):
    third = next((n for n in others if n != a and n != b), None)
    if third is None:
        return a < b
    sorted_three = sort_three([a, b, third])
    return sorted_three[0] == a or (sorted_three[1] == a and sorted_three[2] == b) or (sorted_three[2] == a and sorted_three[1] == b)

def merge(left, right, all_numbers):
    merged = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        if compare_two(left[i], right[j], all_numbers):
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged

def merge_sort_optimized(data):
    n = len(data)
    if n <= 1:
        return data
    if n <= 3:
        return sort_three(data)

    mid = n // 2
    left = merge_sort_optimized(data[:mid])
    right = merge_sort_optimized(data[mid:])
    return merge(left, right, data)

def sort_with_three_number_choice_optimized():
    global choice_count
    numbers = list(range(1, 99))
    print("Starting optimized sorting process...")
    sorted_numbers = merge_sort_optimized(numbers)
    print("\nFinal determined sorting (optimized):", sorted_numbers)
    print("用户选择（三数排序）的次数:", choice_count)

if __name__ == "__main__":
    sort_with_three_number_choice_optimized()