# Where retrieval fails

The examples below come from the ranked traces for the 100-question local benchmark. Query IDs refer to [the labels](../eval/labels.jsonl). The benchmark tests a fixed collection of public documents and a small number of topics.

## The parser can lose the target before ranking starts

Eight questions have no matching gold chunk in the 1,125-chunk section index. All eight have matches in the fixed-window index. Two concern the breast-milk storage table, three concern formula preparation, and three concern feeding techniques in the WIC guide. For q037, hybrid retrieval puts the correct CDC document and page first, but the chunk starts after the table containing the requested time limits. For q082 it finds the WIC page on readiness for solids but misses the labeled sitting skill. A document-and-page match would count both as successes.

A useful next experiment would preserve short numbered steps and serialize table rows with their headers, then measure whether the eight gold snippets become reachable. The current section chunker drops short bodies and detects headings heuristically. Fixed windows retain more text but can put adjacent sections into the same chunk.

## Topic and age are insufficient to identify a passage

For q021, hybrid retrieval ranks a sleep brochure first for a question about the direction of rolling on the six-month milestone checklist. The brochure discusses rolling and its age metadata includes the requested band. For q033, the first result comes from the correct toddler checklist, but it describes motor skills rather than imitating household chores. Metadata filtering narrows the candidate set while leaving the ranker to distinguish the specific skill.

Broad document-level age tags also limit the interpretation of age-band correctness. It tests overlap with stored metadata. It does not certify that the passage is appropriate for a particular child.

## Gold coverage and source coverage limit the scores

The index contains passages from nine of the eleven PDFs. One small file is a publisher redirect notice, and one image-based handout supplies too little text without OCR. The USDA WIC guide supplies 943 of the 1,125 section chunks. One document can therefore dominate both rankings and the kinds of question the collection supports.

The labels mostly name one acceptable source. A useful passage in a different brochure can be scored as a miss. The questions were assembled from known documents, and several share a checklist page. The aggregate measures performance on this specific set. Independent parent-written questions and adjudication of other relevant passages would make the comparison more informative.

## Answer quality

The recorded run exercised retrieval only. Its scores cannot establish factual accuracy or faithfulness of generated answers. The judge and refusal controls are available for a keyed run. The keyless demo exposes retrieved text and its source page so readers can inspect what the model would receive.
