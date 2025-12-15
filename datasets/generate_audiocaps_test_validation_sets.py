from datasets import load_dataset
from tqdm import tqdm


import torch
from torch.utils.data import DataLoader, RandomSampler



def collate_fn(batch):
    max_len = max(len(item["audio"]["array"]) for item in batch)  # Find max audio length

    for item in batch:
        audio_tensor = torch.tensor(item["audio"]["array"])
        if len(audio_tensor) < max_len:
            # Pad with zeros
            pad_size = max_len - len(audio_tensor)
            audio_tensor = torch.cat([audio_tensor, torch.zeros(pad_size)], dim=0)
        else:
            # Truncate
            audio_tensor = audio_tensor[:max_len]

        # Replace the audio array with the padded tensor
        item["audio"]["array"] = audio_tensor

    # Return structured batch with all keys intact
    return {
        key: ([item[key] for item in batch] if key != "audio" else
              {
                  "path": [item["audio"]["path"] for item in batch],
                  "array": torch.stack([item["audio"]["array"] for item in batch]).unsqueeze(1),
                  "sampling_rate": [item["audio"]["sampling_rate"] for item in batch]  # Keep as list
              })
        for key in batch[0]
    }


def dump_files(dataset_name = 'OpenSound/AudioCaps', split = 'test' , dump_dir = 'datasets/audiocaps_test/', num_srcs = 3, duration_hrs= 5.11):
    ds = load_dataset(dataset_name)

    dataset = ds[split].shuffle(seed=42)
    
    sampler = RandomSampler(dataset, replacement=False)
    dataloader = DataLoader(dataset, batch_size= num_srcs, sampler = sampler, collate_fn= collate_fn)

    hrs = 0

    srs = set()
    cnt = 0


    for batch in tqdm(dataloader):

        starts = batch['start_time']

        annots= batch['caption']

        sampling_rates = batch['audio']['sampling_rate']

        audio = batch['audio']['array']


        audio_lengths = batch['audio_length']



        max_audio_length = max(audio_lengths)

        time = max_audio_length/ sampling_rates[0]

        hrs += time/3600

        torch.save(batch, f'{dump_dir}dat_{cnt}.pt')

        if hrs > duration_hrs:

            break


        cnt += 1


if __name__ == '__main__':

    #dump_files(dataset_name = 'OpenSound/AudioCaps', split = 'test' ,
    #            dump_dir = 'datasets/audiocaps_test/', num_srcs = 3, duration_hrs= 5.11)
    

    dump_files(dataset_name = 'OpenSound/AudioCaps', split = 'validation' ,
                dump_dir = 'datasets/audiocaps_valid/', num_srcs = 3, duration_hrs= 3.96)
