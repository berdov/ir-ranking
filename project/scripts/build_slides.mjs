#!/usr/bin/env node
// Editable native PowerPoint evidence, authored with the supplied Artifact Tool.
// Usage: RUNTIME_NODE .../scripts/build_slides.mjs [presentation-data.json] [final.pptx]
// The checked-in PPTX is the deliverable; rebuilding needs the artifact runtime.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';

const ROOT=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const RUNTIME=process.env.CODEX_PRIMARY_RUNTIME_DIR ?? path.join(os.homedir(),'.cache/codex-runtimes/codex-primary-runtime/dependencies');
const MODULES=process.env.RUNTIME_NODE_MODULES ?? path.join(RUNTIME,'node/node_modules');
const PYTHON=process.env.RUNTIME_PYTHON ?? path.join(RUNTIME,'python/bin/python3');
const SKILL=process.env.PRESENTATIONS_SKILL_DIR ?? path.join(os.homedir(),'.codex/plugins/cache/openai-primary-runtime/presentations/26.1119.11507/skills/presentations');
process.env.RUNTIME_NODE_MODULES=MODULES;
process.env.RUNTIME_NODE=process.execPath;
process.env.RUNTIME_PYTHON=PYTHON;
process.env.RUNTIME_BIN_DIR=path.join(RUNTIME,'bin/override');
process.env.PATH=process.env.RUNTIME_BIN_DIR+path.delimiter+process.env.PATH;
const requireFromRuntime=createRequire(path.join(MODULES,'__report_runtime__.cjs'));
const {Presentation,PresentationFile}=await import(pathToFileURL(requireFromRuntime.resolve('@oai/artifact-tool')).href);
const {finalizePresentation,applyPresentationChartFont,resolvePresentationFont}=await import(pathToFileURL(path.join(SKILL,'container_tools/artifact_tool_utils.mjs')).href);
const FONT=resolvePresentationFont({fontFamily:'Arial'});
const COLOR={ink:'#172B3A',muted:'#536373',accent:'#1B6E62',blue:'#627B9B',gray:'#A4A4A4',paper:'#FFFFFF'};
const SMOKE=process.argv.includes('--smoke');
const INPUT=process.argv[2]&&!SMOKE?path.resolve(process.argv[2]):path.join(ROOT,'reports/figures/presentation-data.json');
const OUTPUT=process.argv[3]&&!SMOKE?path.resolve(process.argv[3]):path.join(ROOT,SMOKE?'.build/smoke-output/checked.pptx':'slides/final-project.pptx');
const BUILD=path.join(ROOT,'.build',SMOKE?'slides-smoke':'slides');
await fs.mkdir(BUILD,{recursive:true});
await fs.mkdir(path.dirname(OUTPUT),{recursive:true});
const deck=Presentation.create({slideSize:{width:1280,height:720}});
const measured=n=>{if(n==null||!Number.isFinite(Number(n)))throw Error(`Missing or invalid measurement: ${n}`);return Number(n);};
const fmt=(n,d=4)=>measured(n).toFixed(d);
const int=n=>measured(n).toLocaleString('ru-RU');
// Keep the embedded Excel chart snapshot within its decimal precision budget.
// Exact measurements remain in presentation-data.json and source reports.
const chartNumber=n=>Number(measured(n).toFixed(8));

function text(slide,value,left,top,width,height,size=24,bold=false,color=COLOR.ink){
  const obj=slide.shapes.add({geometry:'textbox',position:{left,top,width,height},fill:'none',line:{fill:'none',width:0}});
  obj.text=String(value);
  obj.text.style={typeface:FONT,fontSize:`${size}pt`,bold,color,verticalAlignment:'top',insets:0,autoFit:'none'};
  return obj;
}
function slide(title,notes='',cover=false){
  const s=deck.slides.add();s.background.fill=COLOR.paper;
  text(s,title,64,44,1152,cover?180:86,cover?42:32,true);
  text(s,String(deck.slides.items.length),1180,660,40,25,12,false,COLOR.muted);
  s.speakerNotes.text=notes;
  return s;
}
function table(s,values,{top=170,height=330,width=1152,cols=[650,240,262],size=22}={}){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:64,top,width,height,columnWidths:cols,values});
  t.styleOptions={headerRow:true,bandedRows:false};
  t.borders.assign({style:'solid',fill:'#D6DEE3',width:1});
  for(let r=0;r<values.length;r++)for(let c=0;c<values[r].length;c++){
    const cell=t.getCell(r,c);cell.fill=r===0?'#E9EFF2':'#FFFFFF';
    cell.text.style={typeface:FONT,fontSize:`${size}pt`,bold:r===0,color:COLOR.ink};
  }
  return t;
}
function chart(s,type,config){
  const c=s.charts.add(type,config);applyPresentationChartFont(c,{fontFamily:FONT});return c;
}
function sourceNotes(note,sources=[]){return note+'\n\nИсточники:\n'+sources.join('\n');}

if(SMOKE){
  const s=slide('Ранжирование документов\nVK MARCO','Проверка шрифта и рендеринга. Не итоговая презентация.',true);
  text(s,'Игорь Бердов\nKaggle: whuliss',64,300,1100,150,30);
  const t=slide('Данные проекта','Фактическое число запросов из файлов соревнования.');
  table(t,[['Раздел','Запросы'],['Обучающие запросы','14 543'],['Тестовые запросы','2 000']],{cols:[800,352],height:240});
  const b=slide('Проверка столбчатого графика','Фактические размеры наборов запросов. Только проверка редактируемого графика.');
  chart(b,'bar',{position:{left:64,top:150,width:1152,height:450},categories:['Train','Test'],series:[{name:'Запросы',values:[14543,2000],fill:COLOR.blue}],barOptions:{direction:'bar'},hasLegend:false,dataLabels:{showValue:true,position:'outEnd'}});
  const sc=slide('Проверка точечного графика','Фактические размеры train и test. Только проверка редактируемого графика.');
  chart(sc,'scatter',{position:{left:64,top:150,width:1152,height:450},series:[{name:'Наборы данных',values:[460030,63546],xValues:[14543,2000],marker:{symbol:'circle',size:7},line:{fill:'none',width:0}}],scatterOptions:{style:'marker'},hasLegend:false,xAxis:{title:'Запросы'},yAxis:{title:'Пары'}});
}else{
  const d=JSON.parse(await fs.readFile(INPUT,'utf8'));
  if(!d.holdout||d.holdout.used_for_selection!==false||!d.source_sha256)throw Error('Completed audited evidence is required');
  if(!OUTPUT.startsWith(path.join(ROOT,'.build')+path.sep)&&!d.runtimes.some(r=>r.source==='reports/official/final-runtime.json'))throw Error('Final slides require measured final-runtime.json; pending runtime is allowed only in a private .build draft');
  const assignment=d.external_sources.assignment, model=d.external_sources.model;
  let s=slide('Ранжирование документов\nVK MARCO',sourceNotes('Около 30 секунд. Задача проекта состоит в переранжировании готовых кандидатов каждого запроса. Качество оценивается nDCG@10.',[assignment]),true);
  text(s,'Игорь Бердов\nKaggle: whuliss',64,300,1050,125,30);
  text(s,'Информационный поиск и ранжирование\nФинальный проект, 2026',64,500,1100,100,22,false,COLOR.muted);

  s=slide('Данные и валидация',sourceNotes('Около 50 секунд. Все разбиения проходят по запросам. Fit используется для обучения, tune для выбора всех параметров и веса нейросети. Holdout открывается после фиксации решения. Все кандидаты одного запроса остаются вместе.',[assignment,'reports/official/splits.json']));
  const split=d.split;
  table(s,[['Часть данных','Запросы','Пары'],['Fit',int(split.query_counts.fit),int(split.row_counts.fit)],['Tune',int(split.query_counts.tune),int(split.row_counts.tune)],['Holdout',int(split.query_counts.holdout),int(split.row_counts.holdout)],['Test',int(d.corpus.test_queries),int(d.corpus.test_pairs)]],{height:330});
  text(s,'nDCG@10: gain = 2ʳᵉˡ − 1. Каждый запрос имеет одинаковый вес.',64,540,1152,60,22);
  text(s,`Seed ${split.seed}. ${int(d.corpus.documents)} документов в корпусе.`,64,610,1152,40,18,false,COLOR.muted);

  s=slide('Признаки и модели',sourceNotes('Около 50 секунд. Корпус обрабатывается потоком. Статистики слов берутся из всего выданного корпуса без разметки. Сравниваются сырые признаки и их ранги внутри запроса. QueryId и DocumentId в обучение не попадают.',[assignment,'features.py','ranking.py','reports/official/selection.json']));
  text(s,`${measured(d.feature_count)} базовых признаков; ${measured(d.feature_count)*2} с рангами`,64,160,1152,60,28,true,COLOR.accent);
  text(s,'BM25F по Title, Body и URL, языковые модели, покрытие запроса, фразы и близость слов.',64,240,1152,110,25);
  text(s,'LightGBM LambdaRank и CatBoost YetiRank. Отдельно проверяются ранги признаков внутри запроса и ансамбли.',64,385,1152,120,25);
  text(s,'Параметры и число деревьев выбираются только на tune.',64,565,1152,65,22,false,COLOR.muted);

  s=slide('Сравнение моделей на tune',sourceNotes('Около 45 секунд. Это результаты подбора, поэтому они не являются независимой итоговой оценкой. Зеленый вариант — выбранный на tune. В итоговой проверке используются новые запросы.', ['reports/official/experiments.json','reports/official/selection.json']));
  const cases=d.slide_tune_choices;
  chart(s,'bar',{position:{left:64,top:150,width:1152,height:450},categories:cases.map(x=>x.label),
    series:[{name:'nDCG@10',values:cases.map(x=>chartNumber(x.tune_ndcg)),fill:COLOR.blue,points:cases.map((x,i)=>({idx:i,fill:x.name===d.selected_model.name?COLOR.accent:COLOR.blue}))}],
    barOptions:{direction:'bar',grouping:'clustered',gapWidth:80},hasLegend:false,
    xAxis:{min:0,max:Math.min(1,Math.max(...cases.map(x=>x.tune_ndcg))+.08),numberFormatCode:'0.00',textStyle:{fontSize:18},majorGridlines:{fill:'#DFE4E8',width:1}},
    yAxis:{min:0,max:Math.min(1,Math.max(...cases.map(x=>x.tune_ndcg))+.08),numberFormatCode:'0.00',textStyle:{fontSize:20},majorGridlines:null},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.0000',textStyle:{fontSize:19,fill:COLOR.ink}}});
  text(s,`Выбор по ${int(split.query_counts.tune)} запросам. Holdout не участвует.`,64,625,1152,40,18,false,COLOR.muted);

  const pureNeural=d.neural_trials.find(t=>t.config.neural_weight===1&&t.config.top_k===0&&t.config.target_query_fraction_on_tune===1);
  const top20=d.neural_trials.filter(t=>t.config.top_k===20&&t.config.target_query_fraction_on_tune===1).sort((a,b)=>b.ndcg10-a.ndcg10)[0];
  const budgetNote=top20?` Top-20 всех запросов даёт ${fmt(top20.ndcg10)} на ${fmt(top20.pair_fraction*100,1)} процента пар. Это доля оценок, а не измеренное ускорение.`:'';
  const pureNote=pureNeural?` Чистая нейросеть даёт ${fmt(pureNeural.ndcg10)}, выбранная смесь ${fmt(d.tune_final_ndcg)}.`:'';
  s=slide('Бюджет нейронного ранжирования',sourceNotes('Около 60 секунд. Предобученный multilingual MiniLM используется zero-shot, без дообучения на проекте. Из документа выбирается одно окно с наибольшим покрытием слов запроса. В собственной проверке меняются глубина reranking и доля запросов. Entropy от стандартизованных скоров является эвристикой формы распределения, а не калиброванной вероятностью ошибки. Все варианты выбираются только на tune.'+pureNote+budgetNote+(d.tune_neural_pair_fraction>=1?' Победила обработка всех пар: entropy gate не выбран, экономии в финальной модели нет.':''),[model,'fusion.py','reports/official/fusion-selection.json']));
  text(s,'mMARCO mMiniLM-L12 (cross-encoder): запрос + заголовок + окно, 256 токенов.',64,128,1152,42,18,false,COLOR.muted);
  const neuralSeries=[10,20,0].map((k,i)=>{const t=d.neural_trials.filter(t=>t.config.top_k===k&&t.config.neural_weight>0);return {name:k?`Top ${k}`:'Все кандидаты',values:t.map(x=>chartNumber(x.ndcg10)),xValues:t.map(x=>chartNumber(x.pair_fraction*100)),marker:{symbol:'circle',size:6,fill:[COLOR.blue,'#9B5E48',COLOR.accent][i]},line:{fill:'none',width:0}};}).filter(s=>s.values.length);
  chart(s,'scatter',{position:{left:64,top:180,width:1152,height:385},series:neuralSeries,scatterOptions:{style:'marker'},hasLegend:true,
    legend:{position:'bottom',textStyle:{fontSize:18}},
    xAxis:{title:'Обработанные нейросетью пары, %',min:0,max:105,numberFormatCode:'0',textStyle:{fontSize:17}},
    yAxis:{title:'nDCG@10 на tune',numberFormatCode:'0.000',textStyle:{fontSize:17},majorGridlines:{fill:'#DFE4E8',width:1}}});
  const neuralSummary=pureNeural?`Смесь ${fmt(d.tune_final_ndcg)} / нейросеть ${fmt(pureNeural.ndcg10)}. α=${fmt(d.fusion.neural_weight,2)}, ${fmt(d.tune_neural_pair_fraction*100,0)}% пар.`:`Выбрано: вес ${fmt(d.fusion.neural_weight,2)}, ${d.fusion.top_k?`top ${d.fusion.top_k}`:'все кандидаты'}, ${fmt(d.tune_neural_pair_fraction*100,1)}% пар.`;
  text(s,neuralSummary,64,580,1152,40,22);
  if(top20)text(s,`Top-20 всех запросов: ${fmt(top20.ndcg10)} на ${fmt(top20.pair_fraction*100,1)}% пар.`,64,630,1152,30,18,false,COLOR.muted);

  s=slide('Финальная проверка на holdout',sourceNotes('Около 60 секунд. После выбора модели она переобучается на fit+tune, затем один раз оценивается на 2000 holdout-запросов. Интервалы относятся к разности nDCG по запросам, а не к абсолютным значениям столбцов. Эти числа нельзя напрямую сравнивать с публичным Kaggle baseline, поскольку запросы другие.',['reports/official/final-holdout.json','reports/official/holdout.json']));
  table(s,[['Вариант','nDCG@10'],...d.holdout_rows.map(r=>[r.label,fmt(r.score)])],{cols:[800,352],height:300});
  const delta=d.holdout.vs_bm25f,ci=delta.paired_bootstrap_95pct_ci;
  text(s,`Разница с BM25F: ${delta.mean_difference>=0?'+':''}${fmt(delta.mean_difference)}\n95% bootstrap CI: [${fmt(ci[0])}; ${fmt(ci[1])}]`,64,520,1152,100,24,true,COLOR.accent);

  s=slide('Воспроизводимость и ограничения',sourceNotes('Около 45 секунд. Решение сохраняет исходные данные отдельно от кода, фиксирует seed, версии зависимостей и revision нейросети. Все веса открытые и используются локально. Первое извлечение признаков длительное; повторные запуски используют валидируемый cache. Отчет содержит реальные замеры, а не только экстраполяцию короткого pilot. Финальный запуск измеряется отдельно при уже готовых лексических признаках и локальных нейросетевых весах; его время не означает полную стоимость холодного старта.',[model,'requirements.txt','solution.py','reports/features/runtime.json','reports/official/final-runtime.json']));
  text(s,'./solution.py --submission_file=submission.csv PATH-TO-DATA',64,160,1152,60,22,true);
  text(s,'Признаки, обучение и запись сабмита; пригодный кэш используется повторно.',64,230,1152,60,20,false,COLOR.muted);
  const rt=d.runtimes.find(r=>r.source.includes('features'));
  const finalRuntime=d.runtimes.find(r=>r.source==='reports/official/final-runtime.json');
  if(rt){
    const memory=rt.native_child_peak_rss_bytes==null?'':`, RSS ${fmt(rt.native_child_peak_rss_bytes/2**30,2)} ГиБ`;
    let timing=`Признаки: ${fmt(rt.wall_seconds/60,1)} мин${memory}.`;
    if(finalRuntime){
      const finalMemory=finalRuntime.native_child_peak_rss_bytes==null?'':`, RSS ${fmt(finalRuntime.native_child_peak_rss_bytes/2**30,2)} ГиБ`;
      timing+=`\nCPU с готовыми признаками: ${fmt(finalRuntime.wall_seconds/60,1)} мин${finalMemory}.`;
    }
    else timing+='\nФинальный CPU-запуск: измеряется.';
    text(s,timing,64,305,1152,120,24);
  }
  text(s,d.tune_neural_pair_fraction>=1?'Одно окно может пропустить удалённый фрагмент. В финале нейросеть оценивает все пары; entropy gate не выбран.':'Одно окно может пропустить удалённый фрагмент. При смене распределения запросов бюджет entropy gate может измениться.',64,475,1152,115,24);
  text(s,'CPU по умолчанию; также поддерживаются Apple MPS и CUDA.',64,620,1152,40,18,false,COLOR.muted);

  const kg=d.kaggle?.public_score ?? d.kaggle?.score;
  const kaggleNotes=kg!=null
    ? `Публичный Kaggle nDCG@10: ${fmt(kg,5)}. Публичный BM25F baseline: ${fmt(d.kaggle.baseline_public_score,3)}. Источник результата: скриншот успешного сабмишна от ${d.kaggle.observed_date}. Приватный результат пока неизвестен.`
    : 'Kaggle-результат добавляется после реальной отправки файла.';
  s=slide('Выводы',sourceNotes(`Около 40 секунд. Подвести итог по независимому holdout и стоимости нейронного этапа. ${kaggleNotes}`, ['reports/official/final-holdout.json','reports/official/fusion-selection.json',...(kg!=null?['reports/official/kaggle-result.json']:[])]));
  text(s,`nDCG@10 на holdout: ${fmt(d.holdout.selected_ndcg)}`,64,165,1152,65,30,true,COLOR.accent);
  const neuralDelta=d.holdout.vs_lexical.mean_difference;
  text(s,`Вклад выбранного нейронного этапа относительно LTR: ${neuralDelta>=0?'+':''}${fmt(neuralDelta)}.`,64,280,1152,100,25);
  const wins=d.holdout_win_tie_loss;
  text(s,`По сравнению с BM25F: улучшение на ${int(wins.win)} запросах, ухудшение на ${int(wins.loss)}, без изменения на ${int(wins.tie)}.`,64,420,1152,110,24);
  text(s,kg!=null?`Публичный Kaggle nDCG@10: ${fmt(kg,5)}`:'Публичный результат Kaggle ещё не получен.',64,600,1152,55,22,false,COLOR.muted);
}

const candidate=path.join(BUILD,'candidate.pptx');
await (await PresentationFile.exportPptx(deck)).save(candidate);
for(let index=0;index<deck.slides.items.length;index++){
  const png=await deck.export({slide:deck.slides.items[index],format:'png',scale:1});
  await fs.writeFile(path.join(BUILD,`slide-${String(index+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
}
const result=await finalizePresentation({
  workspaceDir:ROOT,candidatePath:candidate,finalPath:OUTPUT,
  pythonExecutable:PYTHON,
  integrityValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(SKILL,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--cover-role','cover',...(SMOKE?[2]:[2,6]).flatMap(n=>['--require-native-table-slide',String(n)])],
  explicitTotalSlideCount:SMOKE?4:8,
  requiredNativeTableOwnerSlides:SMOKE?[2]:[2,6],requiredNativeChartOwnerSlides:SMOKE?[3,4]:[4,5],
  nativeChartTargetApplication:'portable',materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:[FONT]},verifyArtifactToolImport:true,
  receiptPath:path.join(BUILD,path.basename(OUTPUT,'.pptx')+'-validation.json'),
});
console.log(JSON.stringify({output:OUTPUT,previews:BUILD,slides:deck.slides.items.length,result}));
