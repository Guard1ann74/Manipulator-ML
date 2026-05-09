import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def visualize():
    try:
        with open('model_transform/training_stats.json', 'r', encoding='utf-8') as f:
            stats = json.load(f)
    except FileNotFoundError:
        print("Файла для отображения результатов еще нет. Сначала нужно запустить train_nn.py")
        return

    # Настройка стиля
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(18, 12))
    plt.suptitle(f"РЕЗУЛЬТАТЫ ОБУЧЕНИЯ", fontsize=22, fontweight='bold', y=0.98)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    cv_scores = stats['cv_scores']
    x_range = range(1, 6)
    ax1.plot(x_range, cv_scores, marker='o', markersize=10, color='#2ecc71', linewidth=3, zorder=3)
    ax1.fill_between(x_range, cv_scores, color='#2ecc71', alpha=0.2)
    ax1.axhline(y=stats['cv_mean'], color='red', linestyle='--', label=f'Среднее: {stats["cv_mean"]:.4f}')
    ax1.set_title('Стабильность модели на кросс-валидации', fontsize=16, pad=15)
    ax1.set_ylim(0.85, 1.0)
    ax1.set_xticks(x_range)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    cm = np.array(stats['confusion_matrix'])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                xticklabels=stats['class_names'], yticklabels=stats['class_names'],
                cbar=True, annot_kws={"size": 12, "weight": "bold"})
    ax2.set_title('Матрица ошибок', fontsize=16, pad=15)
    ax2.set_xlabel('Предсказано')
    ax2.set_ylabel('Реально')

    ax3 = fig.add_subplot(gs[1, 0])
    class_names = stats['class_names']
    plot_data = []
    for cls in class_names:
        for m in ['precision', 'recall', 'f1-score']:
            plot_data.append({'Класс': cls, 'Метрика': m, 'Значение': stats['report'][cls][m]})

    df_metrics = pd.DataFrame(plot_data)
    sns.barplot(data=df_metrics, x='Класс', y='Значение', hue='Метрика', ax=ax3, palette='viridis')
    ax3.set_title('Метрики качества по классам', fontsize=16, pad=15)
    ax3.set_ylim(0, 1.1)
    ax3.legend(loc='lower right', ncol=1)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    summary_info = (
        f"ОСНОВНЫЕ ХАРАКТЕРИСТИКИ:\n"
        f"{'-' * 48}\n"
        f"Test Accuracy:       {stats['test_accuracy']:.4%}\n"
        f"CV Mean Accuracy:    {stats['cv_mean']:.4%}\n"
        f"CV Std Deviation:    {np.std(stats['cv_scores']):.4f}\n"
        f"Samples in Test:     {sum(sum(row) for row in cm)}\n"
        f"Total Dataset:       {len(stats['cv_scores']) * (sum(sum(row) for row in cm) // 1)}\n\n"
        f"ПАРАМЕТРЫ МОДЕЛИ:\n"
        f"{'-' * 48}\n"
        f"Embedder:            RuBERT-tiny2 (28.5M params)\n"
        f"Classifier:          SVC"
    )

    ax4.text(-0.03, 0.11,
             summary_info,
             fontsize=16,
             family='monospace',
             verticalalignment='bottom',
             weight='bold',
             bbox=dict(
                 facecolor='white',
                 alpha=0.8,
                 boxstyle='round,pad=0.8',
                 edgecolor='#bdc3c7')
             )

    output_path = 'model_transform/final_training_report.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    visualize()
