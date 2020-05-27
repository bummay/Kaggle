|home|away|stadium|abbr|
|:--|:--|:--|:--|
|横浜Ｆ・マリノス|浦和レッズ|日産スタジアム|nissan|
|横浜Ｆ・マリノス|川崎フロンターレ|ニッパツ三ツ沢球技場|nippatsu|
|横浜ＦＣ|ジェフユナイテッド千葉|ニッパツ三ツ沢球技場|nippatsu|
|横浜ＦＣ|カマタマーレ讃岐|味の素フィールド西が丘|nishigaoka|
|ガンバ大阪|セレッソ大阪|万博記念競技場|bampaku|

list_clubinfoを元にhomeとawayをコードに置き換える。
|home|away|stadium|abbr|
|:--|:--|:--|:--|
|**yokohama_fm**|**urawa**|日産スタジアム|nissan|
|**yokohama_fm**|**kawasaki**|ニッパツ三ツ沢球技場|nippatsu|
|**yokohama_fc**|**chiba**|ニッパツ三ツ沢球技場|nippatsu|
|**yokohama_fc**|**sanuki**|味の素フィールド西が丘|nishigaoka|
|**g_osaka**|**c_osaka**|万博記念競技場|bampaku|

homeとstadiumの組み合わせがlist_clubinfoに存在すれば、abbrにスタジアムのabbrをセット。存在しなければabbrに「other」をセット
|home|away|stadium|abbr|
|:--|:--|:--|:--|
|**yokohama_fm**|**urawa**|日産スタジアム|nissan|
|**yokohama_fm**|**kawasaki**|ニッパツ三ツ沢球技場|nippatsu|
|**yokohama_fc**|**chiba**|ニッパツ三ツ沢球技場|nippatsu|
|**yokohama_fc**|**sanuki**|味の素フィールド西が丘|**other**|
|**g_osaka**|**c_osaka**|万博記念競技場|bampaku|

[home]*[abbr]のカテゴリ変数列を作成する。
|home|away|stadium|abbr|yokohama_fm_nissan|yokohama_fm_nippatsu|yokohama_fc_nippatsu|g_osaka_bampaku|yokohama_fc_other|
|:--|:--|:--|:--|:--:|:--:|:--:|:--:|:--:|
|yokohama_fm|urawa|日産スタジアム|nissan|1|0|0|0|0|
|yokohama_fm|kawasaki|ニッパツ三ツ沢球技場|nippatsu|0|1|0|0|0|
|yokohama_fc|chiba|ニッパツ三ツ沢球技場|nippatsu|0|0|1|0|0|
|yokohama_fc|sanuki|味の素フィールド西が丘|nishigaoka|0|0|0|0|1|
|g_osaka|c_osaka|万博記念競技場|bampaku|0|0|0|**1**|0|

