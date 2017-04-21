package kr.ac.jejunu;

import org.junit.Before;
import org.junit.Test;
import org.springframework.context.ApplicationContext;
import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.support.GenericXmlApplicationContext;

import java.sql.SQLException;
import java.util.Random;

import static org.hamcrest.CoreMatchers.*;
import static org.hamcrest.MatcherAssert.*;

public class ProductDaoTest {

    ProductDao productDao;

    @Before
    public void set() {
        ApplicationContext context = new GenericXmlApplicationContext("daoFactory.xml");
        productDao = context.getBean("productDao", ProductDao.class);
    }

    @Test
    public void get() throws SQLException, ClassNotFoundException {
        Long id = 1L;
        String title = "제주감귤";
        Integer price = 15000;

        Product product = productDao.get(id);
        assertThat(id, is(product.getId()));
        assertThat(title, is(product.getTitle()));
        assertThat(price, is(product.getPrice()));
    }


    @Test
    public void add() throws SQLException, ClassNotFoundException {
        Long id = Long.valueOf(new Random().nextInt());
        String title = "제주감귤";
        Integer price = 15000;

        Product product = new Product();
        product.setId(id);
        product.setTitle(title);
        product.setPrice(price);

        productDao.add(product);
        Product addProduct = productDao.get(id);
        assertThat(id, is(addProduct.getId()));
        assertThat(title, is(addProduct.getTitle()));
        assertThat(price, is(addProduct.getPrice()));
    }


    @Test
    public void update() throws SQLException, ClassNotFoundException {
        Long id = Long.valueOf(new Random().nextInt());
        String title = "제주감귤";
        Integer price = 15000;

        Product product = new Product();
        product.setId(id);
        product.setTitle(title);
        product.setPrice(price);
        productDao.add(product);


        String changedTitle = "맛있";
        Integer changedPrice = 5000;
        product.setId(id);
        product.setTitle(changedTitle);
        product.setPrice(changedPrice);
        productDao.update(product);

        Product updateProduct = productDao.get(id);
        assertThat(id, is(updateProduct.getId()));
        assertThat(changedTitle, is(updateProduct.getTitle()));
        assertThat(changedPrice, is(updateProduct.getPrice()));
    }


    @Test
    public void delete() throws SQLException, ClassNotFoundException {
        Long id = Long.valueOf(new Random().nextInt());
        String title = "제주감귤";
        Integer price = 15000;

        Product product = new Product();
        product.setId(id);
        product.setTitle(title);
        product.setPrice(price);

        productDao.add(product);
        productDao.delete(id);

        Product addProduct = productDao.get(id);
        assertThat(addProduct, nullValue());
    }
}
