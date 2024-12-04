# DataLoader formats

## General format

Used in SHAPE & CLEVR dataloader

For batch size N, and max question length n, max program length p, C channels, THEN DataLoader returns 6-tuple consisting of:

0. Nx(n+2) Tensor of question word indices. Plus 2 due to start/end tokens
1. N Tensor of problem indices (i.e., problem #1, problem #2, ...)
   - NOTE: These indices are created at processing time for SHAPES. Therefore, not necessarily reliable.
   - NOTE: Also created at *processing time* of CLEVR. These are not CLEVR's original indices.
2. NxCxHxW Tensor of images. For images, the colour channel order is RGB
   1. SHAPES: Just the image
   2. CLEVR: Extracted ResNet-101 features. Can be switched to the image
3. N-tuple of scene-graphs, or N-tuple of `None`
   1. Each scene-graph is a list of objects
   2. Each object is a dictionary with keys:
      1. 'id'
      2. 'position'
      3. 'color'
      4. 'material'
      5. 'shape'
      6. 'size'
4. N Tensor of answers; using answer word indices
5. Nx(p+2) Tensor of program word indices, or N-tuple of `None`. Plus 2 due to start/end tokens.
   - NOTE: In prefix notation; e.g. `+ 5 4`.
6. N-tuple of `None`, or N Tensor of question "type" numbers (when available)
   1. CLEVR: Note, this has nothing to do with "question_family_index". This
             is a completely different value created at processing time; effectively
             each root function in the program gets a type.
   2. SHAPES & SHAPES-SyGeT, I generated families myself (these are meaningful), but 
   the dataloader does NOT return the family, it again returns the "type" (I think, again root primitive but not 100% sure)
   see q_to_json scripts for details
7. Question length N tensor; (note, *length*; not index of <END>)
8. Program length: N-tuple of `None` or N tensor; (note, *length*; not index of <END>)


In addition to the above, mapping to/from indices and tokens is
represented by vocab. Vocab is a dictionary with the following keys:

`['question_token_to_idx', 'program_token_to_idx', 'answer_token_to_idx', 'program_token_arity', 'question_idx_to_token', 'program_idx_to_token', 'answer_idx_to_token']`

each key should map to a dictionary; either tokens -> IDS, or IDS -> tokens resp.

### CLEVR

Notes, test split has no scene graphs, answers, program tokens or lengths, 
question family numbers, or program lengths.
