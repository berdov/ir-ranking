#!/usr/bin/env python3
"""Build evidence-only plots and slide input from completed experiment JSON.

No metrics are estimated. Missing final holdout results stop the build. Source
hashes travel with the derived chart/table data so a deck can be rebuilt after
a new Kaggle result without changing any measured validation result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def number(value, name):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f'Nonfinite measurement: {name}')
    return value


def count(value):
    return f'{int(value):,}'.replace(',', ' ')


def nice_name(name):
    if name == 'baseline_bm25f':
        return 'BM25F'
    if name.startswith('blend_'):
        return 'Ансамбль деревьев'
    family = 'CatBoost' if 'catboost' in name else ('LightGBM' if 'lightgbm' in name else name)
    return family + (' + ранги' if name.endswith('_rank') else '')


def save_figure(fig, stem):
    fig.tight_layout()
    fig.savefig(stem.with_suffix('.png'), dpi=180, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def collect(root, output_dir):
    official = root/'reports'/'official'
    sources = {name: official/name for name in (
        'experiments.json', 'selection.json', 'splits.json', 'fusion-selection.json',
        'holdout.json', 'final-holdout.json')}
    absent = [str(p) for p in sources.values() if not p.is_file()]
    if absent:
        raise FileNotFoundError('Final reports are not ready: ' + ', '.join(absent))
    reports = {name: load_json(path) for name, path in sources.items()}
    exp, selected, split = (reports[name] for name in ('experiments.json','selection.json','splits.json'))
    fusion, holdout = (reports[name] for name in ('fusion-selection.json','final-holdout.json'))
    if exp.get('quick_smoke_only') or selected.get('quick_smoke_only'):
        raise ValueError('Smoke-test numbers cannot be published as project results')
    if holdout.get('used_for_selection') is not False:
        raise ValueError('The report must identify the final holdout as report-only')
    feature_archive = root/'cache'/'features.npz'
    with np.load(feature_archive, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        feature_count = len(archive['feature_names'])
    if not metadata.get('complete'):
        raise ValueError('Partial corpus profiles cannot be presented as final data')
    choices = [{**case, 'label': nice_name(case['name'])} for case in exp['choices']]
    # All individual tree trials plus the actually selected ensemble, if any.
    slide_choices = [case for case in choices if not case['name'].startswith('blend_') or case['name']==selected['name']]
    trials = []
    for trial in fusion['trials']:
        item = dict(trial)
        item['ndcg10'] = number(item['ndcg10'], 'tune nDCG')
        item['pair_fraction'] = int(item['neural_pairs']) / int(fusion['tune_pairs'])
        trials.append(item)
    final_rows = [
        {'label':'BM25F','score':number(holdout['bm25f_ndcg'],'BM25F holdout')},
        {'label':'Лексический LTR','score':number(holdout['lexical_ndcg'],'lexical holdout')},
        {'label':'Выбранный конвейер','score':number(holdout['selected_ndcg'],'final holdout')},
    ]
    qids = sorted(holdout['per_query_selected'])
    if set(qids) != set(holdout['per_query_bm25f']) or set(qids) != set(holdout['per_query_lexical']):
        raise ValueError('Per-query holdout scores are not aligned')
    deltas = np.array([holdout['per_query_selected'][q]-holdout['per_query_bm25f'][q] for q in qids])
    config = fusion['config']
    runtimes = []
    for path in sorted((root/'reports').rglob('*runtime*.json')):
        obj = load_json(path)
        if 'wall_seconds' in obj:
            runtimes.append({'source':str(path.relative_to(root)),**obj})
            sources[str(path.relative_to(root))] = path
    kaggle = None
    kaggle_path = official/'kaggle-result.json'
    if kaggle_path.exists():
        kaggle = load_json(kaggle_path)
        sources['kaggle-result.json'] = kaggle_path
    for relative in ('reports/official/query-cases.json', 'reports/linux-smoke.json',
                     'reports/linux-full-feature-comparison.json', 'reports/linux-native-selection-hashes.json',
                     'reports/linux-refit-comparison.json', 'reports/linux-fusion-comparison.json',
                     'reports/linux-numeric-portability-probe.json', 'reports/linux-portability-check.json',
                     'reports/linux-runtime-selection-portability.json', 'reports/linux-neural-smoke.json',
                     'reports/official/final-code-check.json', 'reports/official/portable-cli-replay.json'):
        path = root/relative
        if path.exists():
            sources[relative] = path
    summary = {
        'title':'Ранжирование документов VK MARCO', 'author':'Игорь Бердов', 'kaggle_username':'whuliss',
        'corpus':metadata, 'feature_count':feature_count,
        'split':{'seed':split['seed'],'query_counts':split['query_counts'],'row_counts':split['row_counts']},
        'selected_model':selected, 'tune_choices':choices, 'slide_tune_choices':slide_choices,
        'neural_trials':trials, 'fusion':config, 'fusion_details':fusion,
        'tune_final_ndcg':fusion['tune_ndcg10'],
        'tune_neural_pair_fraction':fusion['selected_neural_pairs']/fusion['tune_pairs'],
        'holdout_rows':final_rows, 'holdout':holdout,
        'holdout_win_tie_loss':{'win':int((deltas>1e-12).sum()),'tie':int((np.abs(deltas)<=1e-12).sum()),'loss':int((deltas< -1e-12).sum())},
        'runtimes':runtimes, 'kaggle':kaggle,
        'source_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in sources.items()},
        'external_sources':{
            'assignment':'https://github.com/RuslanAllayarov/ir-course-hse-2026/tree/main/project',
            'model':'https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825'},
    }
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'presentation-data.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary, deltas


def build_plots(summary, deltas, output_dir):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    cases = summary['slide_tune_choices']
    values = [case['tune_ndcg'] for case in cases]
    labels = [case['label'] for case in cases]
    fig, ax = plt.subplots(figsize=(9,4.8))
    colors = ['#1b6e62' if case['name']==summary['selected_model']['name'] else '#627b9b' for case in cases]
    bars=ax.barh(labels,values,color=colors)
    ax.set(xlabel='nDCG@10 на tune',title='Сравнение вариантов на 2 000 обучающих запросов')
    ax.set_xlim(0, min(1,max(values)+.075)); ax.invert_yaxis()
    ax.bar_label(bars,fmt='%.4f',padding=4)
    save_figure(fig,output_dir/'tune-models')

    fig, ax = plt.subplots(figsize=(8,5))
    trials = summary['neural_trials']
    baseline = next(t['ndcg10'] for t in trials if t['config']['neural_weight']==0)
    colors={0:'#1b6e62',10:'#627b9b',20:'#9b5e48'}
    for k in (10,20,0):
        group=[t for t in trials if t['config']['top_k']==k and t['config']['neural_weight']>0]
        if group:
            ax.scatter([100*t['pair_fraction'] for t in group],[t['ndcg10'] for t in group],
                       label='Все кандидаты' if k==0 else f'Первые {k}',color=colors[k],alpha=.7,s=35)
    ax.axhline(baseline,color='#555555',linestyle='--',label='Лексический LTR')
    ax.scatter([100*summary['tune_neural_pair_fraction']],[summary['tune_final_ndcg']],
               color='#111111',marker='*',s=180,label='Выбранная настройка',zorder=5)
    ax.set(xlabel='Доля пар, обработанных нейросетью, %',ylabel='nDCG@10 на tune',
           title='Качество и объём нейронного вычисления')
    ax.legend(fontsize=9); ax.grid(alpha=.15)
    save_figure(fig,output_dir/'neural-budget')

    fig, ax = plt.subplots(figsize=(8,4.5))
    rows=summary['holdout_rows']
    bars=ax.bar([r['label'] for r in rows],[r['score'] for r in rows],color=['#a4a4a4','#627b9b','#1b6e62'])
    ax.set(ylim=(0,min(1,max(r['score'] for r in rows)+.08)),ylabel='nDCG@10',title='Финальная проверка на 2 000 отложенных запросов')
    ax.bar_label(bars,fmt='%.4f',padding=4)
    save_figure(fig,output_dir/'holdout-comparison')

    fig, ax=plt.subplots(figsize=(8,4.5))
    ax.hist(deltas,bins=40,color='#627b9b',edgecolor='white')
    ax.axvline(0,color='#222222',linewidth=1)
    ax.set(xlabel='Изменение nDCG@10 относительно BM25F',ylabel='Число запросов',
           title='Распределение улучшений и ошибок на holdout')
    save_figure(fig,output_dir/'query-gains')


def update_report(root, summary):
    """Replace only explicitly generated result blocks in the authored report."""
    path = root/'REPORT.md'
    if not path.exists():
        return
    report = path.read_text(encoding='utf-8')

    def replace(name, body):
        nonlocal report
        start, end = f'<!-- {name}_START -->', f'<!-- {name}_END -->'
        if report.count(start) != 1 or report.count(end) != 1:
            raise ValueError(f'Missing or ambiguous report block: {name}')
        first, rest = report.split(start)
        _, last = rest.split(end)
        report = first + start + '\n' + body.strip() + '\n' + end + last

    replace('FIGURES_TUNE', '![Сравнение моделей на tune](reports/figures/tune-models.png)\n\n'
            'Для компактности на графике показаны отдельные модели; все проверенные смеси приведены в таблице выше.')
    f = summary['fusion_details']
    config = summary['fusion']
    grouped = {}
    for trial in summary['neural_trials']:
        c = trial['config']
        if c['neural_weight'] == 0:
            continue
        key = (c['top_k'], c['target_query_fraction_on_tune'])
        if key not in grouped or trial['ndcg10'] > grouped[key]['ndcg10']:
            grouped[key] = trial
    lines = ['| Глубина | Целевая доля запросов | Лучший α | nDCG@10 | Нейросетевые пары |',
             '| --- | ---: | ---: | ---: | ---: |',
             f"| Только LTR | 0% | 0 | {f['lexical_ndcg10']:.6f} | 0 |"]
    for (k, fraction), trial in sorted(grouped.items(), key=lambda item: (item[0][0] or 10**6, item[0][1])):
        lines.append(f"| {k or 'Все'} | {fraction:.0%} | {trial['config']['neural_weight']:.2f} | "
                     f"{trial['ndcg10']:.6f} | {count(trial['neural_pairs'])} ({trial['pair_fraction']:.1%}) |")
    lines += ['', f"Выбрано: **α={config['neural_weight']:.2f}, top_k={config['top_k'] or 'все'}, "
              f"entropy_threshold={config['entropy_threshold']:.6f}**. Tune nDCG@10 — "
              f"**{f['tune_ndcg10']:.6f}**, разница с выбранным лексическим LTR "
              f"**{f['tune_ndcg10']-f['lexical_ndcg10']:+.6f}**.",
              f"Для этой конфигурации нужны {count(f['selected_neural_pairs'])} из {count(f['tune_pairs'])} пар "
              f"({summary['tune_neural_pair_fraction']:.1%}) на {count(f['selected_neural_queries'])} из "
              f"{count(f['tune_queries'])} запросов.", '',
              '![Качество и объём нейронного вычисления](reports/figures/neural-budget.png)', '',
              'Каждая точка — отдельный вариант сетки; звезда — выбранная настройка. '
              'Это сравнение количества необходимых нейросетевых оценок, а не прямой замер ускорения. '
              'Для подбора были заранее вычислены нейросетевые оценки всех tune-пар; '
              'накладные расходы и размер батча не позволяют переводить долю пар в точное ускорение.']
    pure = next((t for t in summary['neural_trials'] if t['config']['neural_weight']==1
                 and t['config']['top_k']==0 and t['config'].get('target_query_fraction_on_tune')==1), None)
    if pure:
        lines += ['', f"**Чистая нейросеть** (α=1, все кандидаты всех запросов): nDCG@10 "
                  f"**{pure['ndcg10']:.6f}**. Выбранная смесь отличается от неё на "
                  f"**{f['tune_ndcg10']-pure['ndcg10']:+.6f}** на tune."]
    budget = grouped.get((20, 1.0))
    if budget:
        lines += ['', f"При ограничении **top-20 для всех запросов** лучший вариант даёт "
                  f"**{budget['ndcg10']:.6f}** на {budget['pair_fraction']:.1%} пар. "
                  f"Отставание от выбранной конфигурации — {f['tune_ndcg10']-budget['ndcg10']:.6f}. "
                  'Это возможный компромисс при ограничении вычислений, но финальная конфигурация выбиралась по максимуму качества.']
    if summary['tune_neural_pair_fraction'] >= 1:
        lines += ['', '**Отрицательный результат идеи экономии:** на tune победила обработка всех пар. '
                  'Entropy gate и ограничение глубины не выбраны; экономию нейросетевых оценок финальному решению не приписываем.']
    replace('NEURAL_RESULTS', '\n'.join(lines))

    h = summary['holdout']
    lines = [f"Конфигурация заморожена до оценки **{count(h['queries'])} запросов / {count(h['pairs'])} пар**. "
             '`used_for_selection=false`. Для обучения деревьев использованы fit+tune.', '',
             '| Вариант | nDCG@10 |', '| --- | ---: |']
    lines += [f"| {r['label']} | {r['score']:.6f} |" for r in summary['holdout_rows']]
    lines += ['', '| Разность | Средняя Δ nDCG@10 | Парный bootstrap, 95% CI |', '| --- | ---: | --- |']
    for key, label in [('vs_bm25f','Выбранный конвейер − BM25F'),('vs_lexical','Выбранный конвейер − лексический LTR')]:
        result = h[key]
        lo, hi = result['paired_bootstrap_95pct_ci']
        lines.append(f"| {label} | {result['mean_difference']:+.6f} | [{lo:+.6f}; {hi:+.6f}] |")
    lines += ['', f"Bootstrap использует {count(h['vs_bm25f']['bootstrap_repeats'])} повторов с ресемплингом целых запросов. "
              'Оба варианта в каждом повторе оцениваются на одних и тех же запросах. '
              'Интервал относится к разности метрик; он не учитывает возможный сдвиг данных соревнования.', '',
              '![Итоговое сравнение на holdout](reports/figures/holdout-comparison.png)', '',
              '![Изменение метрики по запросам](reports/figures/query-gains.png)']
    w = summary['holdout_win_tie_loss']
    lines += ['', f"Относительно BM25F качество выросло на {count(w['win'])} запросах, снизилось на {count(w['loss'])}, "
              f"не изменилось на {count(w['tie'])} (допуск 10⁻¹²). Для выбранной настройки нейросеть "
              f"обрабатывает {count(h['neural_pairs'])} пар и {count(h['neural_queries'])} запросов holdout."]
    replace('HOLDOUT_RESULTS', '\n'.join(lines))
    case_report = root/'reports'/'official'/'query-cases.json'
    if case_report.exists():
        cases = load_json(case_report)['cases']
        lines = ['Три запроса выбраны вручную после заморозки модели: выигрыш, ошибка и смешанный результат. '
                 'Они не использовались для подбора и не являются доказательством причин изменения метрики.', '',
                 '| Запрос | BM25F | LTR | Итог | Δ к LTR |', '| --- | ---: | ---: | ---: | ---: |']
        for case in cases:
            metric = case['ndcg10']
            lines.append(f"| {case['query']} | {metric['bm25f']:.6f} | {metric['ltr']:.6f} | "
                         f"{metric['final']:.6f} | {case['deltas']['final_minus_ltr']:+.6f} |")
        for case in cases:
            lines += ['', f"**{case['query']}** (`{case['query_id']}`). {case['observation']}"]
            if case['query_id']=='train:35553':
                lines += ['', 'В текущей лексической токенизации остаются слова от двух символов, '
                          'поэтому одноцифровое «9» отсутствует в термах запроса; cross-encoder получает исходный запрос. '
                          'Это правдоподобная гипотеза для различия на этом примере, но причинность без отдельной абляции не установлена. '
                          'Сохранение одноцифровых чисел — направление дальнейшей работы; зафиксированный токенизатор не менялся.']
        lines += ['', 'Метрика учитывает top-10, а наблюдения выше описывают top-3. '
                  'По этим целенаправленно выбранным примерам нельзя оценить частоту ошибок. '
                  'Полные списки с оценками, заголовками и фрагментами: '
                  '[`query-cases.md`](reports/official/query-cases.md); '
                  'данные и хеши: [`query-cases.json`](reports/official/query-cases.json).']
        replace('QUERY_CASES', '\n'.join(lines))

    runtime = []
    if f.get('neural_metadata'):
        runs = f['neural_metadata'].get('runs', [])
        for index, run in enumerate(runs, 1):
            runtime.append(f"Нейросетевой проход tune №{index}: {count(run['rows'])} пар, "
                           f"{run['seconds']:.2f} с, {run['pairs_per_second']:.2f} пары/с. "
                           f"Устройство `{f['inference_environment']['device']}`, "
                           f"batch_size={f['inference_environment']['batch_size']}.")
    runtime += ['', '| Измеренный этап | Время, с | Пик RSS, ГиБ |', '| --- | ---: | ---: |']
    stage_names = {'reports/features/runtime.json':'Полные признаки, macOS',
                   'reports/official/tuning-runtime.json':'Подбор лексических моделей',
                   'reports/official/neural-tune-runtime.json':'Нейросетевой tune, MPS',
                   'reports/official/audit-runtime.json':'Аудит holdout, MPS',
                   'reports/official/final-runtime.json':'Финальный CPU-запуск, готовые признаки',
                   'reports/linux-full-features-runtime.json':'Полные признаки, Linux',
                   'reports/linux-refit-runtime.json':'Переобучение фиксированного CatBoost, Linux',
                   'reports/linux-neural-smoke-runtime.json':'Нейросетевые 96 пар, Linux CPU'}
    for measurement in summary['runtimes']:
        rss = measurement.get('native_child_peak_rss_bytes')
        memory = f'{rss/2**30:.2f}' if rss is not None else 'не измерен'
        label = stage_names.get(measurement['source'], measurement['source'])
        runtime.append(f"| [{label}]({measurement['source']}) | {measurement['wall_seconds']:.2f} | {memory} |")
    if not any(r['source']=='reports/official/final-runtime.json' for r in summary['runtimes']):
        runtime += ['', '**Финальный CPU-запуск ещё измеряется.** Его полное время и память здесь пока не приведены.']
    runtime += ['', 'Разные строки могут описывать вложенные этапы, поэтому их время не следует механически складывать. '
                'Замеры RSS не включают отдельную память GPU; они относятся к указанной платформе, а не гарантируют такую же скорость на другом CPU.']
    replace('RUNTIME_RESULTS', '\n'.join(runtime))

    linux_comparison = root/'reports'/'linux-full-feature-comparison.json'
    if linux_comparison.exists():
        comparison = load_json(linux_comparison)
        smoke = load_json(root/'reports'/'linux-smoke.json')
        lines = [f"В Linux aarch64 / Python {smoke['python']} / CPU прошли "
                 f"**{smoke['tests_run']} из {smoke['tests_run']}** исходных синтетических/unit-тестов, "
                 'проверки импортов и `pip check`. Затем независимо обработан весь корпус: '
                 f"получена матрица **{count(comparison['shape'][0])} × {comparison['shape'][1]}** "
                 f"`{comparison['dtype']}`. Имена признаков, идентификаторы, метки и статистики корпуса совпали."]
        if comparison['checks']['X_bitwise_equal']:
            lines += ['', 'Числовая матрица побитово совпала с macOS.']
        else:
            changed = comparison['columns_with_bitwise_differences']
            lines += ['', f"**Побитовое сравнение матрицы не прошло:** отличаются {count(comparison['unequal_float32_elements'])} "
                      f"элементов только в {len(changed)} столбцах (`"+'`, `'.join(changed)+'`). '
                      f"Максимальная абсолютная разница — **{comparison['max_absolute_difference']:.3g}**; "
                      f"остальные {comparison['shape'][1]-len(changed)} столбцов совпали. "
                      'Диагностический пример локализовал различие в платформенном `float32 np.log1p` '
                      'для длинового штрафа LM с μ=1500. Исходный отчёт с `success=false` сохранён; '
                      'эту проверку не выдаём за успешное побитовое равенство.']
        lines += ['', '**Проверка происхождения вместо привязки к последним битам.** '
                  '`portability.py` сверяет исходные файлы, сохранённый SHA-256 полного корпуса, '
                  'код извлечения и зафиксированных компонентов, версии NumPy/PyStemmer, схему, '
                  'идентификаторы, метки, статистики и разбиение. Для текущего запуска в копии конфигурации '
                  'пересчитываются только хеши fit/tune. Метод, выбранные гиперпараметры и исходные '
                  '`best_config.json`/`fusion_config.json` не меняются; повторного подбора нет. '
                  'Полный SHA-256 корпуса вычисляется при извлечении; при повторном использовании кэша '
                  'сверяются размер, время изменения и хеши краёв файла. '
                  'Это проверка происхождения штатного кэша или нового извлечения, **не поэлементная '
                  'проверка допуска погрешности X** и не проверка произвольно отредактированной матрицы.']
        refit_path = root/'reports'/'linux-refit-comparison.json'
        if refit_path.exists():
            refit = load_json(refit_path)
            result = refit['linux_refit_linux_features_vs_mac_model_mac_features']
            if result['ranking_identical']:
                lines += ['', f"**Независимое обучение в Linux:** CatBoost с теми же {refit['iterations']} деревьями "
                          f"и seed {refit['seed']} переобучен на всех {count(refit['train_rows'])} строках. "
                          f"Ранжирование всех {count(refit['test_rows'])} тестовых пар совпало с macOS; "
                          f"запросов с другим порядком — {result['queries_with_different_order']}. "
                          f"Максимальная разница лексических скоров — {result['max_absolute_score_difference']:.3g}."]
        fusion_path = root/'reports'/'linux-fusion-comparison.json'
        if fusion_path.exists():
            fused = load_json(fusion_path)
            if fused.get('linux_refit_final_csv_identical'):
                lines += ['', '**Комбинация с переобученной Linux-моделью и теми же ранее вычисленными '
                          'нейросетевыми оценками Mac CPU** дала побитово исходный CSV: '
                          f"`{fused['reference_csv_sha256']}`. Эта проверка подтверждает переносимость "
                          'извлечения, обучения и смешивания, но сама по себе не проверяет повторный '
                          'расчёт всех нейросетевых оценок в Linux.']
        neural_path = root/'reports'/'linux-neural-smoke.json'
        if neural_path.exists():
            neural = load_json(neural_path)
            if neural.get('success') and neural.get('neural_ranks_identical'):
                lines += ['', f"**Настоящий нейросетевой расчёт на Linux CPU:** заново обработаны "
                          f"{neural['rows']} пар из {len(neural['query_ids'])} полных тестовых групп "
                          f"({', '.join(neural['query_ids'])}). Порядок нейросети и результат смешивания "
                          f"на этих группах совпали с macOS; максимальная разница логитов — "
                          f"{neural['max_abs_logit_difference']:.3g}. Это проверка подмножества, не всех "
                          'тестовых запросов.']
        assessment_path = root/'reports'/'linux-portability-check.json'
        if assessment_path.exists():
            assessment = load_json(assessment_path)
            if assessment.get('success'):
                lines += ['', f"После исправления совместимости прошли **{assessment['unit_tests']} Linux-тестов**, "
                          'проверка происхождения полного кэша и строгая проверка повторного использования '
                          'модели в `fit_final` ([итоговая проверка](reports/linux-portability-check.json)).']
        code_check_path = root/'reports'/'official'/'final-code-check.json'
        replay_path = root/'reports'/'official'/'portable-cli-replay.json'
        if code_check_path.exists() and replay_path.exists():
            check, replay = load_json(code_check_path), load_json(replay_path)
            if check.get('cached_cli_replay_after_portability_fix') and check.get('prediction_changed') is False:
                lines += ['', f"На macOS финальные {check['tests_passed']} тестов и повторный CLI-запуск "
                          f"с готовым кэшем ({replay['elapsed_seconds']:.2f} с) подтвердили, что исправление "
                          'совместимости сохранило исходный CSV '
                          '([проверка финального кода](reports/official/final-code-check.json), '
                          '[отчёт повторного запуска](reports/official/portable-cli-replay.json)).']
        lines += ['', '**Граница проверки:** полный пересчёт всех нейросетевых оценок теста '
                  'внутри Linux пока не выполнялся. Полный холодный запуск всего конвейера в Linux '
                  'не заявляется как измеренный. Проверена архитектура aarch64; Linux x86_64 не проверялся.', '',
                  'Источники: [`linux-smoke.json`](reports/linux-smoke.json), '
                  '[`linux-full-feature-comparison.json`](reports/linux-full-feature-comparison.json), '
                  '[`linux-numeric-portability-probe.json`](reports/linux-numeric-portability-probe.json), '
                  '[`linux-refit-comparison.json`](reports/linux-refit-comparison.json), '
                  '[`linux-fusion-comparison.json`](reports/linux-fusion-comparison.json), '
                  '[`linux-neural-smoke.json`](reports/linux-neural-smoke.json), '
                  '[`linux-runtime-selection-portability.json`](reports/linux-runtime-selection-portability.json).']
        replace('LINUX_RESULTS', '\n'.join(lines))

    delta = h['vs_lexical']
    lo, hi = delta['paired_bootstrap_95pct_ci']
    interval = ('Интервал улучшения относительно LTR находится выше нуля.' if lo > 0 else
                'Интервал разности с LTR включает ноль; уверенный положительный эффект на этом holdout не установлен.' if hi >= 0 else
                'Интервал разности с LTR находится ниже нуля: выбранная по tune нейросетевая настройка ухудшила holdout.')
    kg = summary['kaggle'] or {}
    kg_score = kg.get('public_score', kg.get('score'))
    kaggle = ('Публичный Kaggle-скор ещё не получен; победа над официальным бейзлайном 0.592 пока не подтверждена.'
              if kg_score is None else f"Публичный Kaggle nDCG@10 — **{number(kg_score,'Kaggle score'):.5f}**. "
              + ('Официальный бейзлайн 0.592 побит.' if float(kg_score) > .592 else 'Официальный бейзлайн 0.592 пока не побит.'))
    replace('CONCLUSIONS', f"Итоговый holdout nDCG@10 — **{h['selected_ndcg']:.6f}**, "
            f"разница с BM25F **{h['vs_bm25f']['mean_difference']:+.6f}**, с лексическим LTR "
            f"**{delta['mean_difference']:+.6f}**. {interval}\n\n{kaggle}")
    path.write_text(report, encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project_dir',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output_dir',type=Path)
    args=parser.parse_args()
    root=args.project_dir.resolve()
    output_dir=args.output_dir or root/'reports'/'figures'
    summary,deltas=collect(root,output_dir)
    build_plots(summary,deltas,output_dir)
    update_report(root,summary)
    print(json.dumps({'output_dir':str(output_dir),'source_sha256':summary['source_sha256']},ensure_ascii=False))


if __name__=='__main__':
    main()
