# search_comparison.py
# Лабораторная работа №1: Введение в алгоритмы. Сложность. Поиск
# Автор: Студент 3 курса, Программная инженерия
# Дата: 26.01.2026

import timeit
import numpy as np
import matplotlib.pyplot as plt
import platform
import sys
import json
from typing import List, Tuple


# ==================== АЛГОРИТМЫ ПОИСКА ====================

def linear_search(arr: List[int], target: int) -> int:
    """
    Линейный поиск элемента в массиве.

    Алгоритм последовательно проверяет каждый элемент массива
    до тех пор, пока не найдет искомый элемент или не достигнет конца.

    Args:
        arr: Массив для поиска
        target: Искомый элемент

    Returns:
        int: Индекс найденного элемента или -1, если элемент не найден

    Временная сложность:
        - Лучший случай: O(1) - элемент находится в начале массива
        - Средний случай: O(n/2) = O(n) - элемент в середине
        - Худший случай: O(n) - элемент в конце или отсутствует

    Пространственная сложность: O(1) - не требует дополнительной памяти
    """
    for i in range(len(arr)):  # O(n) - проход по всем элементам в худшем случае
        if arr[i] == target:  # O(1) - сравнение двух чисел
            return i  # O(1) - возврат индекса
    return -1  # O(1) - элемент не найден
    # Общая сложность: O(n)


def binary_search(arr: List[int], target: int) -> int:
    """
    Бинарный поиск элемента в отсортированном массиве.

    Алгоритм многократно делит интервал поиска пополам.
    На каждом шаге сравнивает искомый элемент со средним элементом
    и продолжает поиск в левой или правой половине.

    Args:
        arr: Отсортированный массив для поиска
        target: Искомый элемент

    Returns:
        int: Индекс найденного элемента или -1, если элемент не найден

    Предусловия:
        - Массив должен быть отсортирован по возрастанию

    Временная сложность:
        - Лучший случай: O(1) - элемент находится в середине
        - Средний случай: O(log n)
        - Худший случай: O(log n) - элемент в начале/конце или отсутствует

    Пространственная сложность: O(1) - итеративная реализация
    """
    left = 0  # O(1) - инициализация левой границы
    right = len(arr) - 1  # O(1) - инициализация правой границы

    while left <= right:  # O(log n) - цикл выполняется log₂(n) раз
        mid = (left + right) // 2  # O(1) - вычисление среднего индекса

        if arr[mid] == target:  # O(1) - сравнение
            return mid  # O(1) - элемент найден
        elif arr[mid] < target:  # O(1) - сравнение
            left = mid + 1  # O(1) - отбрасываем левую половину
        else:  # O(1)
            right = mid - 1  # O(1) - отбрасываем правую половину

    return -1  # O(1) - элемент не найден
    # Общая сложность: O(log n)
    # Количество итераций: ⌈log₂(n)⌉ (округление вверх)


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def get_pc_info() -> str:
    """Возвращает информацию о характеристиках ПК."""
    return f"""
Характеристики ПК для тестирования:
- Процессор: {platform.processor() or 'Информация недоступна'}
- Архитектура: {platform.machine()}
- ОС: {platform.system()} {platform.release()}
- Python: {sys.version.split()[0]}
"""


def test_algorithms() -> bool:
    """
    Тестирует корректность реализации алгоритмов поиска.

    Returns:
        bool: True, если все тесты пройдены успешно
    """
    print("="*70)
    print("ТЕСТИРОВАНИЕ КОРРЕКТНОСТИ АЛГОРИТМОВ")
    print("="*70)

    test_array = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    test_cases = [
        (1, "первый элемент", 0),
        (19, "последний элемент", 9),
        (9, "средний элемент", 4),
        (20, "отсутствующий элемент", -1),
        (0, "элемент меньше минимального", -1)
    ]

    print(f"\nТестовый массив: {test_array}")
    print("-"*70)
    print(f"{'Элемент':<10} {'Описание':<30} {'Ожидается':<12} {'Linear':<10} {'Binary':<10} {'Статус'}")
    print("-"*70)

    all_passed = True
    for target, description, expected in test_cases:
        linear_result = linear_search(test_array, target)
        binary_result = binary_search(test_array, target)

        passed = (linear_result == expected and binary_result == expected and 
                 linear_result == binary_result)
        status = "✅ OK" if passed else "❌ FAIL"

        if not passed:
            all_passed = False

        print(f"{target:<10} {description:<30} {expected:<12} "
              f"{linear_result:<10} {binary_result:<10} {status}")

    print("-"*70)
    if all_passed:
        print("✓ Все тесты пройдены успешно!\n")
    else:
        print("✗ Обнаружены ошибки в реализации!\n")

    return all_passed


def run_performance_analysis(sizes: List[int] = None, num_runs: int = 20) -> dict:
    """
    Проводит эмпирический анализ производительности алгоритмов поиска.

    Args:
        sizes: Список размеров массивов для тестирования
        num_runs: Количество повторений для усреднения

    Returns:
        dict: Результаты замеров времени выполнения
    """
    if sizes is None:
        sizes = [1000, 5000, 10000, 50000, 100000, 500000, 1000000]

    print("="*70)
    print("ЭМПИРИЧЕСКИЙ АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*70)
    print(f"\nРазмеры массивов: {sizes}")
    print(f"Количество повторений для усреднения: {num_runs}")
    print("\nСценарии поиска:")
    print("  1. Худший случай - элемент в конце массива")
    print("  2. Средний случай - элемент в середине массива")
    print("  3. Отсутствующий элемент")

    results = {
        'linear_worst': [],
        'linear_avg': [],
        'binary_worst': [],
        'binary_avg': []
    }

    print("\n" + "-"*70)
    print(f"{'Размер':<12} {'Linear (конец)':<18} {'Linear (сред.)':<18} "
          f"{'Binary (конец)':<18} {'Binary (сред.)':<18}")
    print(f"{'N':<12} {'мс':<18} {'мс':<18} {'мс':<18} {'мс':<18}")
    print("-"*70)

    for size in sizes:
        arr = list(range(size))  # Отсортированный массив [0, 1, 2, ..., size-1]

        target_end = size - 1
        target_mid = size // 2
        target_not_found = size + 1

        # Линейный поиск - худший случай
        time_linear_worst = timeit.timeit(
            lambda: linear_search(arr, target_end),
            number=num_runs
        ) * 1000 / num_runs
        results['linear_worst'].append(time_linear_worst)

        # Линейный поиск - средний случай
        time_linear_avg = timeit.timeit(
            lambda: linear_search(arr, target_mid),
            number=num_runs
        ) * 1000 / num_runs
        results['linear_avg'].append(time_linear_avg)

        # Бинарный поиск - худший случай
        time_binary_worst = timeit.timeit(
            lambda: binary_search(arr, target_not_found),
            number=num_runs
        ) * 1000 / num_runs
        results['binary_worst'].append(time_binary_worst)

        # Бинарный поиск - средний случай
        time_binary_avg = timeit.timeit(
            lambda: binary_search(arr, target_mid),
            number=num_runs
        ) * 1000 / num_runs
        results['binary_avg'].append(time_binary_avg)

        print(f"{size:<12,} {time_linear_worst:<18.6f} {time_linear_avg:<18.6f} "
              f"{time_binary_worst:<18.6f} {time_binary_avg:<18.6f}")

    print("-"*70)
    print("✓ Замеры завершены успешно!\n")

    results['sizes'] = sizes
    return results


def analyze_complexity(results: dict):
    """
    Анализирует соответствие результатов теоретическим оценкам сложности.

    Args:
        results: Результаты замеров производительности
    """
    sizes = results['sizes']
    linear_worst = results['linear_worst']
    binary_worst = results['binary_worst']

    print("="*70)
    print("АНАЛИЗ СООТВЕТСТВИЯ ТЕОРЕТИЧЕСКИМ ОЦЕНКАМ")
    print("="*70)

    # Проверка O(n) для линейного поиска
    ratio_size = sizes[-1] / sizes[0]
    ratio_time_linear = linear_worst[-1] / linear_worst[0]

    print(f"\nЛинейный поиск (теоретическая сложность O(n)):")
    print("-"*70)
    print(f"Размер увеличился: {sizes[0]:,} → {sizes[-1]:,} ({ratio_size:.1f}x)")
    print(f"Время увеличилось: {linear_worst[0]:.4f} → {linear_worst[-1]:.4f} мс "
          f"({ratio_time_linear:.1f}x)")
    print(f"Ожидаемое увеличение времени для O(n): {ratio_size:.1f}x")
    print(f"Относительная погрешность: "
          f"{abs(ratio_size - ratio_time_linear) / ratio_size * 100:.1f}%")

    # Проверка O(log n) для бинарного поиска
    ratio_log = np.log2(sizes[-1]) / np.log2(sizes[0])
    ratio_time_binary = binary_worst[-1] / binary_worst[0]

    print(f"\nБинарный поиск (теоретическая сложность O(log n)):")
    print("-"*70)
    print(f"log₂({sizes[0]:,}) = {np.log2(sizes[0]):.2f}")
    print(f"log₂({sizes[-1]:,}) = {np.log2(sizes[-1]):.2f}")
    print(f"log₂(N) увеличился в: {ratio_log:.2f}x")
    print(f"Время увеличилось в: {ratio_time_binary:.2f}x")
    print(f"Относительная погрешность: "
          f"{abs(ratio_log - ratio_time_binary) / ratio_log * 100:.1f}%")

    # Сравнение алгоритмов
    speedup = linear_worst[-1] / binary_worst[-1]
    print(f"\nСравнение производительности (N = {sizes[-1]:,}):")
    print("-"*70)
    print(f"Бинарный поиск быстрее линейного в: {speedup:.1f} раз")
    print(f"\nТеоретический анализ операций:")
    print(f"  Линейный поиск: {sizes[-1]:,} операций сравнения")
    print(f"  Бинарный поиск: ~{int(np.log2(sizes[-1]))} операций сравнения")
    print()


def visualize_results(results: dict):
    """
    Создает графики для визуализации результатов.

    Args:
        results: Результаты замеров производительности
    """
    sizes = results['sizes']
    linear_worst = results['linear_worst']
    linear_avg = results['linear_avg']
    binary_worst = results['binary_worst']
    binary_avg = results['binary_avg']

    # График 1: Линейный масштаб
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(sizes, linear_worst, 'r-o', linewidth=2, markersize=6, 
             label='Linear (худший)', alpha=0.8)
    ax1.plot(sizes, linear_avg, 'b-s', linewidth=2, markersize=6, 
             label='Linear (средний)', alpha=0.8)
    ax1.plot(sizes, binary_worst, 'g-^', linewidth=2, markersize=6, 
             label='Binary (худший)', alpha=0.8)
    ax1.plot(sizes, binary_avg, 'm-d', linewidth=2, markersize=6, 
             label='Binary (средний)', alpha=0.8)

    ax1.set_xlabel('Размер массива (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Время (мс)', fontsize=12, fontweight='bold')
    ax1.set_title('Сравнение алгоритмов поиска', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax1.ticklabel_format(style='plain', axis='x')

    # Только бинарный поиск
    ax2.plot(sizes, binary_worst, 'g-^', linewidth=2, markersize=7, 
             label='Binary (худший)', alpha=0.8)
    ax2.plot(sizes, binary_avg, 'm-d', linewidth=2, markersize=7, 
             label='Binary (средний)', alpha=0.8)

    ax2.set_xlabel('Размер массива (N)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Время (мс)', fontsize=12, fontweight='bold')
    ax2.set_title('Бинарный поиск: O(log n)', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    ax2.ticklabel_format(style='plain', axis='x')

    plt.tight_layout()
    plt.savefig('search_comparison_linear.png', dpi=300, bbox_inches='tight')
    print("✓ График сохранен: search_comparison_linear.png")

    # График 2: Логарифмический масштаб
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.semilogy(sizes, linear_worst, 'r-o', linewidth=2, markersize=6, 
                 label='Linear (худший)', alpha=0.8)
    ax1.semilogy(sizes, binary_worst, 'g-^', linewidth=2, markersize=6, 
                 label='Binary (худший)', alpha=0.8)

    ax1.set_xlabel('Размер массива (N)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Время (мс) [log scale]', fontsize=12, fontweight='bold')
    ax1.set_title('Log-scale по Y', fontsize=13, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)

    # Log-log с теоретическими кривыми
    ax2.loglog(sizes, linear_worst, 'r-o', linewidth=2, markersize=6, 
               label='Linear (худший)', alpha=0.8)
    ax2.loglog(sizes, binary_worst, 'g-^', linewidth=2, markersize=6, 
               label='Binary (худший)', alpha=0.8)

    # Теоретические кривые
    x_theory = np.array(sizes)
    y_linear = (linear_worst[0] / sizes[0]) * x_theory
    y_log = (binary_worst[0] / np.log2(sizes[0])) * np.log2(x_theory)

    ax2.loglog(x_theory, y_linear, 'r--', linewidth=1.5, 
               label='O(n) теория', alpha=0.5)
    ax2.loglog(x_theory, y_log, 'g--', linewidth=1.5, 
               label='O(log n) теория', alpha=0.5)

    ax2.set_xlabel('Размер (N) [log scale]', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Время (мс) [log scale]', fontsize=12, fontweight='bold')
    ax2.set_title('Log-log масштаб', fontsize=13, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('search_comparison_log.png', dpi=300, bbox_inches='tight')
    print("✓ График сохранен: search_comparison_log.png")

    plt.show()


def main():
    """Главная функция программы."""
    print("="*70)
    print("ЛАБОРАТОРНАЯ РАБОТА №1")
    print("Тема: Введение в алгоритмы. Сложность. Поиск")
    print("="*70)
    print(get_pc_info())

    # Тестирование корректности
    if not test_algorithms():
        print("Ошибка: тесты не пройдены. Завершение программы.")
        return

    # Эмпирический анализ
    results = run_performance_analysis()

    # Анализ сложности
    analyze_complexity(results)

    # Визуализация
    visualize_results(results)

    # Сохранение результатов
    with open('search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Результаты сохранены в search_results.json")

    print("\n" + "="*70)
    print("РАБОТА ЗАВЕРШЕНА УСПЕШНО!")
    print("="*70)


if __name__ == "__main__":
    main()
