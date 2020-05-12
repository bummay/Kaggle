## Summary

|group|rowcount|
|:--|--:|
|train|1,169,042|
|test|1,295,086|

### columns
|colName|isNa|train_nUnique|train_naCount|test_nUnique|test_naCount|value|
|:--|:--:|--:|--:|--:|--:|:--|
|I1|False|7|0|7|0|0~6|
|I2|False|24|0|24|0|0~23|
|C1|False|9|0|9|0|9種類のString|
|C2|False|171|0|178|0|167種類は共通<br>上位5件(3874378935, 1862037199, 2589684548, 1088910726, 1537671376。trainとtestの両方で5%以上)のみ考慮する。|
|C3|False|347|0|347|0|331種類は共通<br>上位5件(2448089184 , 1892769125 , 98956388 , 3260269773 , 1998340283。trainとtestの両方で5%以上)のみ考慮する。|
|C4|False|7|0|7|0|7種類のString|
|C5|False|223|0|224|0|
|C6|False|16|0|16|0|16種類のString|
|I3|False|3|0|3|0|0~2|
|I4|False|3|0|3|0|0~3|
|I5|False|276|0|318|0|
|I6|False|2|0|2|0|0~1|
|I7|False|2|0|2|0|0~1|
|I8|False|2|0|2|0|0~1|
|I9|False|2|0|2|0|0~1|
|I10|False|2|0|2|0|0~1|
|I11|True|13,353|187,288<br>(16.02%)|13,455|
|I12|True|1,493|187,288<br>(16.02%)|1,412
|I13|True|546,297|187,288<br>(16.02%)|569,177
|I14|True|206|187,288<br>(16.02%)|192
### train
rowCount:1,169,042
|col|nunique|isna|naCount|plan|
|:--|--:|--:|--:|:--|
|I1|7|False|0|たぶん影響なし|
|I2|24|False|0|-|
|C1|9|False|0|-|
|C2|171|False|0|-|
|C3|347|False|0|-|
|C4|7|False|0|-|
|C5|223|False|0|-|
|C6|16|False|0|-|
|I3|3|False|0|-|
|I4|3|False|0|-|
|I5|276|False|0|-|
|I6|2|False|0|-|
|I7|2|False|0|-|
|I8|2|False|0|-|
|I9|2|False|0|-|
|I10|2|False|0|-|
|I11|13,353|True|187,288<br>(16.02%)|-|
|I12|1,493|True|187,288<br>(16.02%)|-|
|I13|546,297|True|187,288<br>(16.02%)|-|
|I14|206|True|39,620<br>(3.39%)|-|

### test
rowCount:1,295,086
|col|nunique|isna|naCount|plan|
|:--|--:|--:|--:|:--|
|I1|7|False|0|たぶん影響なし|
|I2|24|False|0|-|
|C1|9|False|0|-|
|C2|178|False|0|-|
|C3|347|False|0|-|
|C4|7|False|0|-|
|C5|224|False|0|-|
|C6|16|False|0|-|
|I3|3|False|0|-|
|I4|3|False|0|-|
|I5|318|False|0|-|
|I6|2|False|0|-|
|I7|2|False|0|-|
|I8|2|False|0|-|
|I9|2|False|0|-|
|I10|2|False|0|-|
|I11|13,455|True||-|
|I12|1,412|True||-|
|I13|569,177|True||-|
|I14|192|True||-|


### Task
- 各colにおける値ごとのClickedの割合を知りたい。
- nuniqueが多すぎる項目をどうするか。
- I11～I14の欠損値をどう埋めるか。
- I11～I13の欠損件数が同じ187,288件
    → ひょっとして「I11～I13全て欠損 or Not」でFlag化できる？