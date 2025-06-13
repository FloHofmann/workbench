#[cfg(test)]
mod tests {
    use super::*;
    use super::py_interface::retrieve_py_data;
    use std::path::Path;

    #[test]
    fn retrive_py_instance() {
        let datapath = Path(../../data/Data6.mat);
        let instance = retrieve_py_data(datapath);
        let raw_data = instance.getattr("raw_data")?;
        let ch1 = raw_data.get_item("Ch1")?;
        let title = ch1.get_item("title");
        assert_eq!(title, "voltage")
    }
}
