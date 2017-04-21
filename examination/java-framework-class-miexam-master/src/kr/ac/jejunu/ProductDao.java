package kr.ac.jejunu;

import org.springframework.dao.DataAccessException;
import org.springframework.jdbc.core.JdbcTemplate;

import java.sql.*;

public class ProductDao {


    JdbcTemplate jdbcTemplate;

    public ProductDao() {

    }

    public void setJdbcTemplate(JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }



    public Product get(Long id) throws ClassNotFoundException, SQLException {
        String sql = "select * from product where id = ?";
        Object[] params = new Object[]{id};
        Product product = null;

        try {
            product = jdbcTemplate.queryForObject(sql, params, (resultSet, i) -> {
                Product product1 = new Product();
                product1.setId(resultSet.getLong("id"));
                product1.setTitle(resultSet.getString("title"));
                product1.setPrice(resultSet.getInt("price"));

                return product1;
            });
        } catch (DataAccessException e) {
            e.printStackTrace();
        }

        return product;
    }



    public void add(Product product) throws ClassNotFoundException, SQLException {
        String sql = "insert into product (id, title, price) VALUES (?,?,?)";
        Object[] params = new Object[]{product.getId(), product.getTitle(), product.getPrice()};

        jdbcTemplate.update(sql, params);
    }



    public void update(Product product) throws ClassNotFoundException, SQLException {
        String sql = "update product set title = ?, price = ? where id = ?";
        Object[] params = new Object[]{product.getTitle(), product.getPrice(), product.getId()};

        jdbcTemplate.update(sql, params);
    }

    public void delete(Long id)  throws ClassNotFoundException, SQLException {
        String sql = "delete from product where id = ?";
        Object[] params = new Object[]{id};
        jdbcTemplate.update(sql, params);
    }


}
