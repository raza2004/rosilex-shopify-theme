# Evaluation photos

One folder per person, with two or more photos of that person in each:

```
eval/faces/
  person_01/
    a.jpg
    b.jpg
  person_02/
    a.jpg
    b.jpg
    c.jpg
```

Every photo needs exactly one clear face. JPEG, PNG or WebP, up to 8 MB.

`calibrate.py` compares photos of the same person against photos of different people to choose
the pass threshold, so it needs at least two people and at least one person with two photos.
`run_eval.py` applies every style to every photo.

Only use photos you have the right to use for this purpose. This folder is ignored by git.
