# tests

## augment_batch()

end of the function

```cpp
for (int i = 0; i < batch.aug_size; i++) {
    if (batch.aug_targets[i].item().toInt() != batch.aug_samples[i][0][0][0].item().toInt()) {
        std::cout << "fail it " << i << std::endl;
        std::cout << "label " << batch.aug_targets[i].item().toInt() << std::endl;
        std::cout << "value " << batch.aug_samples[i][0][0][0].item().toInt() << std::endl;
        ASSERT(false);
    }
}
```

## get_remote_samples()

inside `for (size_t r = 0; r < num_samples_per_representative; r++)` loop

```cpp
ASSERT(tensor[0][0][0].item().toInt() == i);
```

inside `for (const torch::Tensor& tensor : repr)` loop

```cpp
ASSERT(label == tensor[0][0][0].item().toInt());
DBG("value " << tensor[0][0][0].item().toInt());
```

after `ASSERT(samples.size() == metadata.size());`

```cpp
int s = 0;
DBG("checking if the metadata reflects the segments, iterating on metadata (FAILING)..");
for (auto const &it : metadata) {
    auto label = std::get<0>(it);
    DBG("for label " << label);
    for (int num_reps = 0; num_reps < std::get<2>(it); num_reps++) {
        DBG("num_reps " << num_reps);
        DBG("value " << torch::from_blob(segments[s].first, representative_shape, torch::kFloat32)[0][0][0].item().toInt());
        ASSERT(label == torch::from_blob(segments[s].first, representative_shape, torch::kFloat32)[0][0][0].item().toInt());
        s++;
    }
}
```