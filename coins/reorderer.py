import os


# ----------- ✅ CONFIGURE PATHS -----------
train_dir = "/home/noah/projects/image-identifier/data/coins/digitsannotations/date_cropmatchedout1"
test_dir = "/home/noah/projects/image-identifier/data/coins/digitsannotations/date_cropmatchedout2"


# ----------- ✅ READ DIGIT ORIENTATIONS FROM classes.txt -----------


def read_digit_order(classes_txt_path):
    with open(classes_txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 10 and all(d in "0123456789" for d in lines), "Classes.txt must contain 10 digits"
    return lines


train_digits = read_digit_order(os.path.join(train_dir, "classes.txt"))
test_digits  = read_digit_order(os.path.join(test_dir, "classes.txt"))


# ----------- ✅ BUILD REVERSE MAPS: old_id → digit, then digit → new_id -----------


# For each old_id, find the digit it represented in that folder
train_old_to_digit = {i: digit for i, digit in enumerate(train_digits)}
test_old_to_digit  = {i: digit for i, digit in enumerate(test_digits)}


# Standardized map: digit → correct new class ID
digit_to_new_id = {str(i): i for i in range(10)}


# ----------- ✅ FUNCTION TO REMAP LABEL FILES -----------


def remap_labels(label_dir, old_to_digit_map):
    count = 0
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt") and filename != "classes.txt":
            path = os.path.join(label_dir, filename)
            with open(path, "r") as f:
                lines = f.readlines()


            updated_lines = []
            for line in lines:
                tokens = line.strip().split()
                if not tokens:
                    continue


                old_id = int(tokens[0])
                digit = old_to_digit_map.get(old_id)
                new_id = digit_to_new_id.get(digit)


                if new_id is not None:
                    tokens[0] = str(new_id)
                    updated_lines.append(" ".join(tokens))
                else:
                    print(f"⚠️ Couldn't remap ID {old_id} → digit {digit} in {filename}")


            # Overwrite with remapped labels
            with open(path, "w") as f:
                f.write("\n".join(updated_lines) + "\n")
            count += 1


    print(f"✅ Remapped {count} label files in {label_dir}")


# ----------- ✅ RUN REMAPPING -----------


remap_labels(train_dir, train_old_to_digit)
remap_labels(test_dir, test_old_to_digit)


# ----------- ✅ Overwrite classes.txt with canonical order -----------


canonical = [str(i) for i in range(10)]
for d in [train_dir, test_dir]:
    with open(os.path.join(d, "classes.txt"), "w") as f:
        f.write("\n".join(canonical) + "\n")


print("\n🎯 Training and testing folders now use standardized class IDs: 0–9")
